import io
import os
import random
import warnings
from copy import deepcopy
from enum import StrEnum, auto
from functools import partial
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from chemprop import data
from chemprop.data import ReactionDatapoint
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
from chemprop.featurizers.molgraph import CondensedGraphOfReactionFeaturizer, RxnMode
from chemprop.models.model import MPNN
from chemprop.nn import (
    Aggregation,
    BondMessagePassing,
    RegressionFFN,
    UnscaleTransform,
    agg,
)
from chemprop.nn import metrics as _Metrics
from chemprop.nn.message_passing.base import _MessagePassingBase
from chemprop.nn.utils import Activation
from chemprop.schedulers import build_NoamLike_LRSched
from joblib import Parallel, cpu_count, delayed
from lightning import pytorch as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    model_validator,
)
from rdkit import Chem
from rdkit.Chem.rdchem import Bond, BondType
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from sklearn.preprocessing import StandardScaler
from torch import Tensor, optim
from typing_extensions import Self

from dative_chemprop.data.collate import BatchMolGraph, TrainingBatch
from dative_chemprop.data.dataloader import build_dataloader
from dative_chemprop.data.datapoints import DativeReactionDatapoint
from dative_chemprop.featurizers.atom import DativeAtomFeaturizer


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Aggregation(StrEnum):
    """Aggregation function for message passing."""

    MEAN = auto()
    NORM = auto()
    SUM = auto()


class Metric(StrEnum):
    """Metric for evaluating model performance."""

    RMSE = auto()
    MAE = auto()
    MSE = auto()
    R2SCORE = auto()
    MVELOSS = auto()


def get_agg(aggregation: Aggregation, out_size: int | None = None) -> agg.Aggregation:
    match aggregation:
        case Aggregation.MEAN:
            return agg.MeanAggregation()
        case Aggregation.NORM:
            return agg.NormAggregation()
        case Aggregation.SUM:
            return agg.SumAggregation()
        case _:
            raise ValueError(f"Invalid aggregation function: {aggregation}")


def get_metric(metric: Metric) -> _Metrics.ChempropMetric:
    match metric:
        case Metric.RMSE:
            return _Metrics.RMSE()
        case Metric.MAE:
            return _Metrics.MAE()
        case Metric.MSE:
            return _Metrics.MSE()
        case Metric.R2SCORE:
            return _Metrics.R2Score()
        case Metric.MVELOSS:
            return _Metrics.MVELoss()
        case _:
            raise ValueError(f"Invalid metric: {metric}")


class GraphMAE(pl.LightningModule):
    def __init__(
        self,
        message_passing: _MessagePassingBase,
        d_v: int,
        decoder_hidden_size: int = 256,
        mask_ratio: float = 0.3,
        replace_ratio: float = 0.1,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["message_passing"])
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
            }
        )

        self.message_passing = message_passing
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(message_passing.output_dim, decoder_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_hidden_size, d_v),
        )
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio
        self.mask_token_ratio = 1 - self.replace_ratio
        self.metrics = partial(sce_loss, alpha=2)
        self.enc_mask_token = torch.nn.Parameter(torch.zeros(1, d_v))

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def criterion(self) -> _Metrics.ChempropMetric:
        return self.metrics

    def random_node_masking(self, x: torch.Tensor, mask_ratio: float):
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_ratio * num_nodes)

        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_ratio > 0:
            num_noise_nodes = int(self.replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[
                perm_mask[: int(self.mask_token_ratio * num_mask_nodes)]
            ]
            noise_nodes = mask_nodes[
                perm_mask[-int(self.replace_ratio * num_mask_nodes) :]
            ]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
                :num_noise_nodes
            ]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = []
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        return out_x, mask_nodes, keep_nodes

    def forward(self, bmg: BatchMolGraph):
        # 随机掩码
        out_x, mask_nodes, keep_nodes = self.random_node_masking(bmg.V, self.mask_ratio)
        now_device = out_x.device

        masked_data = bmg.clone()
        masked_data.to(now_device)
        masked_data.V = out_x

        h = self.message_passing(masked_data)
        # 节点重建
        node_recon = self.decoder(h)
        l = self.criterion(node_recon[mask_nodes], bmg.V[mask_nodes])
        return l

    def training_step(self, batch: TrainingBatch, batch_idx: int):
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        l = self(bmg)
        self.log(
            "train_loss",
            l.mean(),
            prog_bar=True,
            on_epoch=True,
            batch_size=len(batch[0]),
        )
        return l

    def validation_step(self, batch: TrainingBatch, batch_idx: int):
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        h = self(bmg)
        node_recon = self.decoder(h)
        l = self.criterion(node_recon, bmg.V)
        self.log(
            "val_loss", l.mean(), prog_bar=True, on_epoch=True, batch_size=len(batch[0])
        )
        return l

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            warnings.warn(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(
                f"Could not find hyper parameters and/or state dict in {path}."
            )

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing",)
            if key not in submodules
        }
        return submodules, state_dict, hparams

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        **kwargs,
    ) -> MPNN:
        submodules = {
            k: v for k, v in kwargs.items() if k in ["message_passing", "decoder"]
        }
        submodules, state_dict, hparams = cls._load(
            checkpoint_path, map_location, **submodules
        )
        kwargs.update(submodules)
        d = torch.load(checkpoint_path, map_location)
        d["state_dict"] = state_dict
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(
            buffer, map_location, hparams_file, strict, **kwargs
        )


class DativeCGR(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str = Field(
        default="DativeCGR",
        description="Name of the model. Determines the directory in which the model is saved and loaded.",
    )
    save_dir: str = Field(
        default=".checkpoints",
        description="Directory in which to save model checkpoints.",
    )
    pretrain_weights: str | None = Field(
        default=None,
        description="Path to pre-trained weights for the message passing layers.",
    )
    seed: int = Field(
        default=42,
        description="Seed for random number generation.",
    )
    max_clip: float | None = Field(
        default=100, description="Maximum value for clipping."
    )
    min_clip: float | None = Field(default=0, description="Minimum value for clipping.")
    cgr_type: (
        Literal[
            "reac_prod",
            "reac_prod_balance",
            "reac_diff",
            "reac_diff_balance",
            "prod_diff",
            "prod_diff_balance",
        ]
        | RxnMode
    ) = Field(
        default=RxnMode.REAC_PROD,
        description="Type of CGR to use. Available options: REAC_PROD, REAC_PROD_BALANCE, "
        "REAC_DIFF, REAC_DIFF_BALANCE, PROD_DIFF, PROD_DIFF_BALANCE.",
    )

    atom_featurizer: MultiHotAtomFeaturizer = Field(
        default=DativeAtomFeaturizer.v1(),
        description="Featurizer for atoms.",
    )
    bond_featurizer: MultiHotBondFeaturizer = Field(
        default=MultiHotBondFeaturizer(
            bond_types=[
                BondType.SINGLE,
                BondType.DOUBLE,
                BondType.TRIPLE,
                BondType.AROMATIC,
                BondType.DATIVE,
            ]
        ),
        description="Featurizer for bonds.",
    )
    hidden_size: int = Field(
        default=300,
        description="Dimensionality of hidden layers.",
    )
    bias: bool = Field(
        default=False,
        description="Whether to include a bias term in the output layer.",
    )
    depth: int = Field(
        default=5,
        description="Number of message passing steps.",
    )
    dropout: float = Field(
        default=0.0,
        description="Dropout probability.",
    )
    undirected: bool = Field(
        default=False,
        description="Whether to treat bonds as undirected.",
    )
    batch_norm: bool = Field(
        default=True,
        description="Whether to use batch normalization in message passing layers.",
    )
    activation: (
        Literal["RELU", "LEAKYRELU", "PRELU", "TANH", "SELU", "ELU"] | Activation
    ) = Field(
        default=Activation.RELU,
        description="Activation function for message passing layers.",
    )
    aggregation: Literal["MEAN", "NORM", "SUM", "ATTENTIVE"] | Aggregation = Field(
        default=Aggregation.NORM,
        description="Aggregation function for message passing.",
    )
    metrics: list[Literal["RMSE", "MAE", "MSE", "R2SCORE", "MVELOSS"] | Metric] = Field(
        default=[Metric.RMSE, Metric.MAE, Metric.R2SCORE],
        description="List of metrics to evaluate model performance.",
    )

    _featurizer: CondensedGraphOfReactionFeaturizer = PrivateAttr(None)
    _message_passing: _MessagePassingBase = PrivateAttr(None)
    _aggregation: agg.Aggregation = PrivateAttr(None)
    _metrics: list[_Metrics.ChempropMetric] = PrivateAttr(None)
    _mpnn: MPNN = PrivateAttr(None)
    _trainer: pl.Trainer = PrivateAttr(None)

    @computed_field
    @property
    def model_path(self) -> str:
        return os.path.join(self.save_dir, f"{self.model_name}_{self.cgr_type}.pt")

    @computed_field
    @property
    def pretrain_path(self) -> str:
        if self.pretrain_weights is None:
            return os.path.join(
                self.save_dir, f"{self.model_name}_{self.cgr_type}_pretrain.pt"
            )
        else:
            return self.pretrain_weights

    def freeze(self) -> None:
        self._mpnn.message_passing.apply(lambda module: module.requires_grad_(False))
        self._mpnn.message_passing.eval()

    def unfreeze(self) -> None:
        self._mpnn.message_passing.apply(lambda module: module.requires_grad_(True))
        self._mpnn.message_passing.train()

    @model_validator(mode="after")
    def _build_layers(self) -> Self:
        setup_seed(self.seed)
        self._featurizer = CondensedGraphOfReactionFeaturizer(
            atom_featurizer=self.atom_featurizer,
            bond_featurizer=self.bond_featurizer,
            mode_=self.cgr_type,
        )
        self._message_passing = BondMessagePassing(
            d_v=self._featurizer.shape[0],
            d_e=self._featurizer.shape[1],
            d_h=self.hidden_size,
            bias=self.bias,
            depth=self.depth,
            dropout=self.dropout,
            activation=self.activation,
            undirected=self.undirected,
        )
        if os.path.exists(self.pretrain_path):
            self._message_passing.load_state_dict(
                {
                    key.replace("message_passing.", ""): value
                    for key, value in torch.load(self.pretrain_path)[
                        "state_dict"
                    ].items()
                    if "message_passing." in key
                }
            )
            print(f"Loaded pre-trained model from {self.pretrain_path}.")
        out_size = self._message_passing.output_dim
        self._aggregation = get_agg(self.aggregation, out_size)
        self._metrics = [get_metric(metric) for metric in self.metrics]
        if os.path.exists(self.model_path):
            self._mpnn = MPNN.load_from_checkpoint(self.model_path)
            print(f"Loaded model from {self.model_path}.")
        return self

    def pretrain(
        self,
        smis: Iterable[str],
        num_workers: int = cpu_count() - 1,
        accelerator="auto",
        devices="auto",
        max_epochs=150,
    ) -> None:
        setup_seed(self.seed)
        all_data = [DativeReactionDatapoint.from_smi(smi) for smi in smis]
        dataset = data.ReactionDataset(all_data, self._featurizer)
        pre_train_model = GraphMAE(
            message_passing=self._message_passing, d_v=self._featurizer.atom_fdim
        )
        pretrain_loader = build_dataloader(
            dataset, num_workers=num_workers, seed=self.seed
        )
        trainer = pl.Trainer(
            logger=False,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            enable_checkpointing=False,
            enable_model_summary=True,
            enable_progress_bar=True,
        )
        trainer.fit(pre_train_model, pretrain_loader)
        trainer.save_checkpoint(self.pretrain_path)

    def fit(
        self,
        smis: Iterable[str],
        y: Iterable[float] | Iterable[list[float]],
        freeze: bool = False,
        batch_size: int = 64,
        num_workers: int = cpu_count() - 1,
        accelerator="auto",
        devices="auto",
        max_epochs=600,
        *,
        val_smis: Iterable[str] = None,
        val_y: Iterable[float] | Iterable[list[float]] = None,
    ) -> None:
        setup_seed(self.seed)
        assert len(smis) == len(y), "X and y must have the same length."
        all_data = [
            DativeReactionDatapoint.from_smi(smi, _y) for smi, _y in zip(smis, y)
        ]
        dataset = data.ReactionDataset(all_data, self._featurizer)
        if val_smis is not None and val_y is not None:
            assert len(val_smis) == len(val_y), "X and y must have the same length."
            val_data = [
                DativeReactionDatapoint.from_smi(smi, _y)
                for smi, _y in zip(val_smis, val_y)
            ]
            val_dataset = data.ReactionDataset(val_data, self._featurizer)
        else:
            val_dataset = None
        if self._mpnn is not None:
            scaler = StandardScaler()
            scaler.mean_ = self._mpnn.predictor.output_transform.mean.cpu().numpy()
            scaler.scale_ = self._mpnn.predictor.output_transform.scale.cpu().numpy()
            dataset.normalize_targets(scaler)
            if val_dataset is not None:
                val_dataset.normalize_targets(scaler)
        else:
            scaler = dataset.normalize_targets()
            if val_dataset is not None:
                val_dataset.normalize_targets(scaler)
            output_transform = UnscaleTransform.from_standard_scaler(scaler)
            ffn = RegressionFFN(
                n_tasks=len(y[0]),
                input_dim=self._message_passing.output_dim,
                output_transform=output_transform,
            )
            self._mpnn = MPNN(
                self._message_passing,
                self._aggregation,
                ffn,
                self.batch_norm,
                self._metrics,
            )

        train_loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=self.seed,
            shuffle=True,
        )
        if val_dataset is not None:
            val_loader = build_dataloader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=self.seed,
                shuffle=False,
            )
        else:
            val_loader = None
        if freeze:
            self.freeze()
        trainer = pl.Trainer(
            logger=False,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True,
        )
        trainer.fit(self._mpnn, train_loader, val_dataloaders=val_loader)
        trainer.save_checkpoint(self.model_path)

    def predict(
        self,
        smis: Iterable[str],
        num_workers: int = cpu_count() - 1,
        accelerator="auto",
        devices="auto",
    ) -> np.ndarray:
        assert self._mpnn is not None, "Model is not trained."
        all_data = Parallel(
            return_as="list",
            n_jobs=num_workers,
            pre_dispatch="1.5*n_jobs",
        )(delayed(DativeReactionDatapoint.from_smi)(smi) for smi in smis)
        dataset = data.ReactionDataset(all_data, self._featurizer)
        test_loader = data.build_dataloader(
            dataset, num_workers=num_workers, shuffle=False
        )
        trainer = pl.Trainer(
            logger=False,
            accelerator=accelerator,
            devices=devices,
            enable_checkpointing=False,
            enable_model_summary=True,
            enable_progress_bar=True,
            inference_mode=True,
        )
        with torch.no_grad():
            test_preds = np.concatenate(trainer.predict(self._mpnn, test_loader))
        return np.clip(test_preds, self.min_clip, self.max_clip)

    def vectorize(
        self,
        smis: Iterable[str],
        num_workers: int = cpu_count() - 1,
        no_conditions: bool = False,
    ) -> np.ndarray:
        assert self._mpnn is not None, "Model is not trained."
        if no_conditions:
            smiles_list = []
            for smi in smis:
                rxn = ReactionFromSmarts(smi)
                smiles_list.append(
                    ".".join([Chem.MolToSmiles(r) for r in rxn.GetReactants()])
                    + ">>"
                    + ".".join([Chem.MolToSmiles(p) for p in rxn.GetProducts()])
                )
        else:
            smiles_list = smis
        all_data = Parallel(
            return_as="list",
            n_jobs=num_workers,
            pre_dispatch="1.5*n_jobs",
        )(delayed(DativeReactionDatapoint.from_smi)(smi) for smi in smiles_list)
        dataset = data.ReactionDataset(all_data, self._featurizer)
        test_loader = data.build_dataloader(
            dataset, num_workers=num_workers, shuffle=False
        )
        with torch.no_grad():
            fingerprints = [
                self._mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=0)
                for batch in test_loader
            ]
            fingerprints = torch.cat(fingerprints, 0)
        self._mpnn = MPNN.load_from_checkpoint(self.model_path)
        return fingerprints.numpy()

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> None:
        self._mpnn = MPNN.load_from_checkpoint(checkpoint_path)
        print(f"Loaded model from {checkpoint_path}.")

    @classmethod
    def load(cls, model_path: str, new_name: str = None, seed: int = 42) -> Self:
        save_dir = os.path.dirname(os.path.abspath(model_path))
        model_name = os.path.basename(model_path).replace(".pt", "")
        for cgr_type in RxnMode.values():
            if model_name.endswith(cgr_type):
                model_name = model_name[: -len(cgr_type)-1]
                break
        else:
            raise ValueError(f"Invalid cgr_type: {cgr_type}.")
        if new_name is not None:
            model = cls(
                model_name=new_name, save_dir=save_dir, seed=seed, cgr_type=cgr_type
            )
            model.load_from_checkpoint(model_path)
        else:
            model = cls(
                model_name=model_name, save_dir=save_dir, seed=seed, cgr_type=cgr_type
            )
        return model

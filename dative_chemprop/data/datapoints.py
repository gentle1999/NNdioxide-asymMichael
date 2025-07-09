'''
Author: TMJ
Date: 2025-02-10 09:43:20
LastEditors: TMJ
LastEditTime: 2025-02-22 16:38:39
Description: 请填写简介
'''
from __future__ import annotations

from dataclasses import dataclass

from chemprop.data.datapoints import _DatapointMixin, _ReactionDatapointMixin
from chemprop.utils import make_mol
from rdkit import Chem
from rdkit.Chem import rdChemReactions


@dataclass
class _DativeReactionDatapointMixin(_ReactionDatapointMixin):

    @classmethod
    def from_smi(
        cls,
        rxn_or_smis: str | tuple[str, str],
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        **kwargs,
    ):
        match rxn_or_smis:
            case str():
                # No longer relies on “>” to determine components, compatible with coordination bonds
                rxn = rdChemReactions.ReactionFromSmarts(rxn_or_smis)
                rct_smi = ".".join(
                    [Chem.MolToSmiles(m) for m in rxn.GetReactants()]
                    + [Chem.MolToSmiles(m) for m in rxn.GetAgents()]
                )
                pdt_smi = ".".join([Chem.MolToSmiles(m) for m in rxn.GetProducts()])
                name = rxn_or_smis
            case tuple():
                rct_smi, pdt_smi = rxn_or_smis
                name = ">>".join(rxn_or_smis)
            case _:
                raise TypeError(
                    "Must provide either a reaction SMARTS string or a tuple of reactant and"
                    " a product SMILES strings!"
                )

        rct = make_mol(rct_smi, keep_h, add_h)
        pdt = make_mol(pdt_smi, keep_h, add_h)

        kwargs["name"] = name if "name" not in kwargs else kwargs["name"]

        return cls(rct, pdt, *args, **kwargs)


@dataclass
class DativeReactionDatapoint(_DatapointMixin, _DativeReactionDatapointMixin):

    def __post_init__(self):
        if self.rct is None:
            raise ValueError("Reactant cannot be `None`!")
        if self.pdt is None:
            raise ValueError("Product cannot be `None`!")

        return super().__post_init__()

    def __len__(self) -> int:
        return 2

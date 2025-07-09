"""
Author: TMJ
Date: 2025-05-30 14:59:57
LastEditors: TMJ
LastEditTime: 2025-05-31 23:20:03
Description: 请填写简介
"""

import itertools
from typing import Literal, Sequence

from pydantic import Field

from dative_chemprop.models.DativeCGR import DativeCGR
from dative_chemprop.utils import *

NAMED_LIGAND = {
    "L3-PiAd": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)N[C@]12CC3C[C@@H](C2)CC(C1)C3)N[C@]12CC3C[C@@H](C2)CC(C1)C3",
    "L3-PiBn": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)NCc1ccccc1)NCc1ccccc1",
    "L3-PiCPh2": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)NC(c1ccccc1)c1ccccc1)NC(c1ccccc1)c1ccccc1",
    "L3-PiCl": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccc(cc1)Cl)Nc1ccc(cc1)Cl",
    "L3-PiEt2": "CCc1cccc(c1NC(=O)[C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(CC)cccc1CC)CC",
    "L3-PiEt2Me": "CCc1cc(C)cc(c1NC(=O)[C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(CC)cc(cc1CC)C)CC",
    "L3-PiEt3": "CCc1cc(CC)cc(c1NC(=O)[C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(CC)cc(cc1CC)CC)CC",
    "L3-PiMe": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccc(cc1)C)Nc1ccc(cc1)C",
    "L3-PiMe2": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(C)cccc1C)Nc1c(C)cccc1C",
    "L3-PiMe2tBu": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(C)cc(cc1C)C(C)(C)C)Nc1c(C)cc(cc1C)C(C)(C)C",
    "L3-PiMe3": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(C)cc(cc1C)C)Nc1c(C)cc(cc1C)C",
    "L3-PiMe4": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(C)c(C)cc(c1C)C)Nc1c(C)c(C)cc(c1C)C",
    "L3-PiMe5": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(C)c(C)c(c(c1C)C)C)Nc1c(C)c(C)c(c(c1C)C)C",
    "L3-PiOEt2": "CCOc1cccc(c1NC(=O)[C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(OCC)cccc1OCC)OCC",
    "L3-PiOMe": "COc1ccc(cc1)NC(=O)[C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccc(cc1)OC",
    "L3-PiOMe2": "COc1cccc(c1NC(=O)[C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(OC)cccc1OC)OC",
    "L3-PiPh": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccccc1)Nc1ccccc1",
    "L3-PiPr2": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(cccc1C(C)C)C(C)C)Nc1c(cccc1C(C)C)C(C)C",
    "L3-PiPr2Ad": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(cc(cc1C(C)C)[C@]12CC3C[C@@H](C2)CC(C1)C3)C(C)C)Nc1c(cc(cc1C(C)C)[C@]12CC3C[C@@H](C2)CC(C1)C3)C(C)C",
    "L3-PiPr2Br": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(cc(cc1C(C)C)Br)C(C)C)Nc1c(cc(cc1C(C)C)Br)C(C)C",
    "L3-PiPr3": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1c(cc(cc1C(C)C)C(C)C)C(C)C)Nc1c(cc(cc1C(C)C)C(C)C)C(C)C",
    "L3-PicH": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)NC1CCCCC1)NC1CCCCC1",
    "L3-PimMe": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccccc1C)Nc1ccccc1C",
    "L3-PimiPr": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccccc1C(C)C)Nc1ccccc1C(C)C",
    "L3-PimtBu": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccccc1C(C)(C)C)Nc1ccccc1C(C)(C)C",
    "L3-PioBr": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccccc1Br)Nc1ccccc1Br",
    "L3-PioMe": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccccc1C)Nc1ccccc1C",
    "L3-PioOMe": "COc1ccccc1NC(=O)[C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccccc1OC",
    "L3-PiotBu": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccccc1C(C)(C)C)Nc1ccccc1C(C)(C)C",
    "L3-PitBu": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)NC(C)(C)C)NC(C)(C)C",
    "L3-PitPM": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)NC(c1ccccc1)(c1ccccc1)c1ccccc1)NC(c1ccccc1)(c1ccccc1)c1ccccc1",
    "L3-PrAd": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)N[C@]12CC3C[C@@H](C2)CC(C1)C3)N[C@]12CC3C[C@@H](C2)CC(C1)C3",
    "L3-PrAd-ent": "O=C([C@H]1CCC[N@+]1([O-1])CCC[N@@+]1([O-1])CCC[C@@H]1C(=O)N[C@]12CC3C[C@@H](C2)CC(C1)C3)N[C@]12CC3C[C@@H](C2)CC(C1)C3",
    "L3-PrBn": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)NCc1ccccc1)NCc1ccccc1",
    "L3-PrCPh2": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)NC(c1ccccc1)c1ccccc1)NC(c1ccccc1)c1ccccc1",
    "L3-PrCl": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1ccc(cc1)Cl)Nc1ccc(cc1)Cl",
    "L3-PrEt2": "CCc1cccc(c1NC(=O)[C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1c(CC)cccc1CC)CC",
    "L3-PrEt2Me": "CCc1cc(C)cc(c1NC(=O)[C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1c(CC)cc(cc1CC)C)CC",
    "L3-PrEt3": "CCc1cc(CC)cc(c1NC(=O)[C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1c(CC)cc(cc1CC)CC)CC",
    "L3-PrMe": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1ccc(cc1)C)Nc1ccc(cc1)C",
    "L3-PrMe2": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1c(C)cccc1C)Nc1c(C)cccc1C",
    "L3-PrMe2Br": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1c(Br)cc(cc1Br)C)Nc1c(Br)cc(cc1Br)C",
    "L3-PrMe3": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1c(C)cc(cc1C)C)Nc1c(C)cc(cc1C)C",
    "L3-PrPh": "O=C([C@@H]1CCCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCCC[C@H]1C(=O)Nc1ccccc1)Nc1ccccc1",
    "L3-PrPr2": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1c(cccc1C(C)C)C(C)C)Nc1c(cccc1C(C)C)C(C)C",
    "L3-PrPr2Br": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1c(cc(cc1C(C)C)Br)C(C)C)Nc1c(cc(cc1C(C)C)Br)C(C)C",
    "L3-PrPr3": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1c(cc(cc1C(C)C)C(C)C)C(C)C)Nc1c(cc(cc1C(C)C)C(C)C)C(C)C",
    "L3-PrcH": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)NC1CCCCC1)NC1CCCCC1",
    "L3-PrcP": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)NC1CCCC1)NC1CCCC1",
    "L3-PrmMe2": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1cc(C)cc(c1)C)Nc1cc(C)cc(c1)C",
    "L3-PrmtBu2": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1cc(cc(c1)C(C)(C)C)C(C)(C)C)Nc1cc(cc(c1)C(C)(C)C)C(C)(C)C",
    "L3-PrmtBu2-ent": "O=C([C@H]1CCC[N@+]1([O-1])CCC[N@@+]1([O-1])CCC[C@@H]1C(=O)Nc1cc(cc(c1)C(C)(C)C)C(C)(C)C)Nc1cc(cc(c1)C(C)(C)C)C(C)(C)C",
    "L3-ProMe": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1ccccc1C)Nc1ccccc1C",
    "L3-ProtBu": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)Nc1ccccc1C(C)(C)C)Nc1ccccc1C(C)(C)C",
    "L3-PrtBu": "O=C([C@@H]1CCC[N@@+]1([O-1])CCC[N@+]1([O-1])CCC[C@H]1C(=O)NC(C)(C)C)NC(C)(C)C",
    "L3-RaAd": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)NC13CC4CC(C3)CC(C1)C4)CCC2)NC12CC3CC(C2)CC(C1)C3",
    "L3-RaBn": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)NCc1ccccc1)CCC2)NCc1ccccc1",
    "L3-RaEt2": "CCc1cccc(c1NC(=O)[C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)Nc1c(CC)cccc1CC)CCC2)CC",
    "L3-RaMe2": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)Nc1c(C)cccc1C)CCC2)Nc1c(C)cccc1C",
    "L3-RaMe2tBu": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)Nc1c(C)cc(cc1C)C(C)(C)C)CCC2)Nc1c(C)cc(cc1C)C(C)(C)C",
    "L3-RaMe3": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)Nc1c(C)cc(cc1C)C)CCC2)Nc1c(C)cc(cc1C)C",
    "L3-RaPh": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)Nc1ccccc1)CCC2)Nc1ccccc1",
    "L3-RaPr2": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)Nc1c(cccc1C(C)C)C(C)C)CCC2)Nc1c(cccc1C(C)C)C(C)C",
    "L3-RaPr3": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)Nc1c(cc(cc1C(C)C)C(C)C)C(C)C)CCC2)Nc1c(cc(cc1C(C)C)C(C)C)C(C)C",
    "L3-RatBu": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)NC(C)(C)C)CCC2)NC(C)(C)C",
    "L3-RatPM": "O=C([C@@H]1C[C@H]2[C@@H]([N@@+]1([O-1])CCC[N@+]1([O-1])[C@H]3CCC[C@H]3C[C@H]1C(=O)NC(c1ccccc1)(c1ccccc1)c1ccccc1)CCC2)NC(c1ccccc1)(c1ccccc1)c1ccccc1",
}

AUGMENT_METHODS = {
    "donor_active": build_rxn_active_smiles,
    "catalyst_complex": build_dative_rxn_smiles,
    "donor_active_catalyst_complex": build_dative_rxn_active_smiles,
}


def get_ligand_smiles(ligand: str) -> str:
    if ligand in NAMED_LIGAND:
        return NAMED_LIGAND[ligand]
    else:
        return ligand


class FengMichaelPredictor(DativeCGR):
    augment_methods: list[
        Literal["donor_active", "catalyst_complex", "donor_active_catalyst_complex"]
    ] = Field(
        default=["donor_active", "catalyst_complex", "donor_active_catalyst_complex"],
    )

    def predict_michael(
        self,
        acceptor: str,
        donor: str,
        product: str,
        salt: str | Sequence[str] | None = None,
        ligand: str | Sequence[str] | None = None,
        solvent: str | Sequence[str] | None = None,
        additive: str | Sequence[str] | None = None,
        temperature: float = 293,
        **kwargs,
    ) -> pd.DataFrame:
        salt, ligand, solvent, additive = [
            ("",) if compound is None else compound
            for compound in (salt, ligand, solvent, additive)
        ]
        salt, ligand, solvent, additive = [
            (compound,) if isinstance(compound, str) else compound
            for compound in (salt, ligand, solvent, additive)
        ]
        ligand = [get_ligand_smiles(lig) for lig in ligand]
        total_combinations = list(itertools.product(salt, solvent, ligand, additive))
        augment_funcs = [build_rxn_smiles] + [
            AUGMENT_METHODS[method] for method in self.augment_methods
        ]

        rxn_smiles_forms = [
            func(
                acceptor=acceptor,
                donor=donor,
                product=product,
                salt=sal,
                solvent=sol,
                ligand=lig,
                additive=add,
            )
            for sal, sol, lig, add in total_combinations
            for func in augment_funcs
        ]
        preds = self.predict(rxn_smiles_forms, **kwargs)
        result_df = pd.DataFrame(
            total_combinations,
            columns=[
                "salt",
                "solvent",
                "ligand",
                "additive",
            ],
        )
        result_df["acceptor"] = acceptor
        result_df["donor"] = donor
        result_df["product"] = product
        result_df["ddG_estimation_mean"] = preds.reshape(-1, 4).mean(axis=1)
        result_df["ddG_estimation_std"] = preds.reshape(-1, 4).std(axis=1)
        result_df["ee_estimation_mean"] = (
            ddG2ee(preds, temperature).reshape(-1, 4).mean(axis=1)
        )
        result_df["ee_estimation_std"] = (
            ddG2ee(preds, temperature).reshape(-1, 4).std(axis=1)
        )
        return result_df

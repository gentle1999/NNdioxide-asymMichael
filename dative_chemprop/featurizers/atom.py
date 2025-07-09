'''
Author: TMJ
Date: 2025-02-19 15:04:48
LastEditors: TMJ
LastEditTime: 2025-02-20 16:06:42
Description: 请填写简介
'''
from typing import Sequence

import numpy as np
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from rdkit.Chem.rdchem import Atom

def is_rare_earth_element(a: Atom) -> bool:
    RARE_EARTH_ELEMENTS = (21, 39, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71)
    if a.GetAtomicNum() in RARE_EARTH_ELEMENTS:
        return True
    else:
        return False

class DativeAtomFeaturizer(MultiHotAtomFeaturizer):
    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        chiral_tags: Sequence[int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[int],
    ):
        self.atomic_nums = {j: i for i, j in enumerate(atomic_nums)}
        self.degrees = {i: i for i in degrees}
        self.formal_charges = {j: i for i, j in enumerate(formal_charges)}
        self.chiral_tags = {i: i for i in chiral_tags}
        self.num_Hs = {i: i for i in num_Hs}
        self.hybridizations = {ht: i for i, ht in enumerate(hybridizations)}

        self._subfeats: list[dict] = [
            self.atomic_nums,
            self.degrees,
            self.formal_charges,
            self.chiral_tags,
            self.num_Hs,
            self.hybridizations,
        ]
        subfeat_sizes = [
            1 + len(self.atomic_nums),
            1 + len(self.degrees),
            1 + len(self.formal_charges),
            1 + len(self.chiral_tags),
            1 + len(self.num_Hs),
            1 + len(self.hybridizations),
            1,
            1,
            1,  # for rare earth element or not
        ]
        self.__size = sum(subfeat_sizes)

    
    def __len__(self) -> int:
        return self.__size

    def __call__(self, a: Atom | None) -> np.ndarray:
        x = np.zeros(self.__size)

        if a is None:
            return x

        feats = [
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            a.GetFormalCharge(),
            int(a.GetChiralTag()),
            int(a.GetTotalNumHs()),
            a.GetHybridization(),
        ]
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()
        x[i + 2] = 1 if is_rare_earth_element(a) else 0
        return x

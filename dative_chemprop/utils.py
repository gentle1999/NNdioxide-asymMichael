import subprocess
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from skfp.distances import dice_count_distance, tanimoto_count_distance
from skfp.fingerprints import ECFPFingerprint

fpgen = ECFPFingerprint(radius=3, count=True)

acceptor_templates_mapping = {
    "Bidentate-1": {
        "template_smarts": "O=[C!R]C=C[C!R]=O",
        "ref_mol_smiles": "CC(/C=C/C(OC(C)(C)C)=O)=O",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A1",
    },
    "Bidentate-2": {
        "template_smarts": "C=C(C=O)C=O",
        "ref_mol_smiles": "COC(=O)/C(=C/c1ccsc1)/C(=O)C",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A2",
    },
    "Bidentate-3": {
        "template_smarts": "C=CC(=O)C(=O)[*]",
        "ref_mol_smiles": "COC(=O)C(=O)/C=C/c1cccc(c1)C",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A3",
    },
    "Bidentate-4": {
        "template_smarts": "C=CC(=O)nn[*]",
        "ref_mol_smiles": "C=CC(=O)n1nc(C)cc1C",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A4",
    },
    "Monodentate-1,2-disub-1": {
        "template_smarts": "[*][C!R]=C[C!R;!$([*#6X2])]=O",
        "ref_mol_smiles": "C/C=C/C=O",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A5",
    },
    "Monodentate-1,2-disub-2": {
        "template_smarts": "[*][C!R]=[C!R][N+](=O)[O-]",
        "ref_mol_smiles": "CC(C)/C=C/[N+]([O-])=O",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A6",
    },
    "Monodentate-1,1-disub-1": {
        "template_smarts": "c1cNC(=O)C1=C",
        "ref_mol_smiles": "CCOC(=O)/C=C/1\\c2ccccc2N(C1=O)C",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A7",
    },
    "Monodentate-1,1-disub-2": {
        "template_smarts": "aC(=O)C(=C)a",
        "ref_mol_smiles": "Clc1cccc(c1)C(=C)C(=O)c1ccccc1",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A8",
    },
    "Monodentate-cyclic-1": {
        "template_smarts": "C=C(C=C1)C=CC1=O",
        "ref_mol_smiles": "O=C1C(C)=C/C(C=C1C)=C\C2=CC=CC=C2",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A9",
    },
    "Monodentate-cyclic-2": {
        "template_smarts": "c1cc(=O)cco1",
        "ref_mol_smiles": "O=C1C=COC2=C1C=CC=C2",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A10",
    },
    "Monodentate-alkyne-1": {
        "template_smarts": "C#CC=O",
        "ref_mol_smiles": "CC(C#C)=O",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A11",
    },
    "Others": {
        "template_smarts": "[*]",
        "ref_mol_smiles": "O=C1C(CCO1)=C",
        "demo_smiles": "",
        "bandwidth": 1,
        "code_name": "A-Others",
    },
}
donor_templates_mapping = {
    "Bidentate-1": {
        "template_smarts": "O=CCC=O",
        "ref_mol_smiles": "O=C(C1)CCCC1=O",
        "demo_smiles": "",
        "charge_atom_index": 2,
        "bandwidth": 1,
        "code_name": "D1",
    },
    "Bidentate-2": {
        "template_smarts": "O=C[CH1!R,CH2!R][#6]=,:[#7]",
        "ref_mol_smiles": "CC(CC(OC)=O)=N",
        "demo_smiles": "",
        "charge_atom_index": 2,
        "bandwidth": 1,
        "code_name": "D2",
    },
    "Monodentate-C-1": {
        "template_smarts": "C[N+]([O-])=O",
        "ref_mol_smiles": "C[N+]([O-])=O",
        "demo_smiles": "",
        "charge_atom_index": 0,
        "bandwidth": 1,
        "code_name": "D3",
    },
    "Monodentate-C-2": {
        "template_smarts": "N#C[CH1,CH2,C-1]",
        "ref_mol_smiles": "N#CCC1=CC=C(C#N)C=C1",
        "demo_smiles": "",
        "charge_atom_index": -1,
        "bandwidth": 1,
        "code_name": "D4",
    },
    "Monodentate-C-3": {
        "template_smarts": "CC=C(C#N)C#N",
        "ref_mol_smiles": "N#C/C(C#N)=C1CCC(C2=CC=CC=C2)CC/1",
        "demo_smiles": "",
        "charge_atom_index": 0,
        "bandwidth": 1,
        "code_name": "D5",
    },
    "Monodentate-C-cyclic-1": {
        "template_smarts": "N1N=C[CH1,CH2]C1=O",
        "ref_mol_smiles": "CC(C(C)=N1)C(N1C2=CC=CC=C2)=O",
        "demo_smiles": "",
        "charge_atom_index": 3,
        "bandwidth": 1,
        "code_name": "D6",
    },
    "Monodentate-C-cyclic-2": {
        "template_smarts": "N1N=CC(=C)C1=O",
        "ref_mol_smiles": "CC(/C1=C/C)=NN(C2=CC=CC=C2)C1=O",
        "demo_smiles": "",
        "charge_atom_index": 3,
        "bandwidth": 1,
        "code_name": "D7",
    },
    "Monodentate-C-cyclic-3": {
        "template_smarts": "c1c[CH1,CH2]C(=O)N1",
        "ref_mol_smiles": "O=C(C1C)NC2=C1C=CC=C2",
        "demo_smiles": "",
        "charge_atom_index": 2,
        "bandwidth": 1,
        "code_name": "D8",
    },
    "Monodentate-C-cyclic-4": {
        "template_smarts": "c1cC(=C)C(=O)N1",
        "ref_mol_smiles": "O=C(NC1=C/2C=CC=C1)C2=C\C",
        "demo_smiles": "",
        "charge_atom_index": 2,
        "bandwidth": 1,
        "code_name": "D9",
    },
    "Monodentate-C-cyclic-5": {
        "template_smarts": "c1cCC(=O)O1",
        "ref_mol_smiles": "O=C(C1C2=CC=CC=C2)OC3=C1C=C(C)C=C3",
        "demo_smiles": "",
        "charge_atom_index": 2,
        "bandwidth": 1,
        "code_name": "D10",
    },
    "Monodentate-C-cyclic-6": {
        "template_smarts": "C1=CCC(=O)O1",
        "ref_mol_smiles": "C1=CCC(=O)O1",
        "demo_smiles": "",
        "charge_atom_index": 2,
        "bandwidth": 1,
        "code_name": "D11",
    },
    "Monodentate-C-cyclic-7": {
        "template_smarts": "c1cC=CC(=O)C1",
        "ref_mol_smiles": "O=C1C(C)C2=CC=CC=C2C=C1",
        "demo_smiles": "",
        "charge_atom_index": 4,
        "bandwidth": 1,
        "code_name": "D12",
    },
    "Monodentate-N-1": {
        "template_smarts": "[#7H2,#7H1]-,:,=[#7]",
        "ref_mol_smiles": "NNC(=O)c1ccccc1",
        "demo_smiles": "",
        "charge_atom_index": 3,
        "bandwidth": 1,
        "code_name": "D13",
    },
    "Monodentate-O-1": {
        "template_smarts": "[OH,$(O[Si])]",
        "ref_mol_smiles": "o1cccc1O[Si](C)(C)C(C)(C)C",
        "demo_smiles": "",
        "charge_atom_index": 1,
        "bandwidth": 1,
        "code_name": "D14",
    },
    "Monodentate-S-1": {
        "template_smarts": "[SH]",
        "ref_mol_smiles": "CS",
        "demo_smiles": "",
        "charge_atom_index": 1,
        "bandwidth": 1,
        "code_name": "D15",
    },
    "Others": {
        "template_smarts": "[*]",
        "ref_mol_smiles": "O=C1OCCC1",
        "demo_smiles": "",
        "charge_atom_index": 0,
        "bandwidth": 1,
        "code_name": "D-Others",
    },
}


def ee2ddG(ee: np.ndarray, temp: np.ndarray | float) -> np.ndarray:
    ddG = np.abs(8.314 * temp * np.log((1 - ee) / (1 + ee)))
    return ddG / 1000 / 4.184  # kJ/mol to kcal/mol


def ddG2ee(ddG: np.ndarray, temp: np.ndarray | float) -> np.ndarray:
    ddG = ddG * 1000 * 4.184  # kcal/mol to kJ/mol
    ee = (1 - np.exp(ddG / (8.314 * temp))) / (1 + np.exp(ddG / (8.314 * temp)))
    return abs(ee)


donor_templates = (
    ReactionFromSmarts("[C:1][*:2]=[*:3][*:4]=[O:5]>>[C:1]=[*:2]-[*:3]=[*:4]-[O-:5]"),
    ReactionFromSmarts("[C-1:1]-[C:2]#[N+:3]-[Si:4]>>[C-:1]-[*:2]#[N+0:3].[Si:4]"),
    ReactionFromSmarts(
        "[#16,#8,#7:1]~[*:2]-[CH2:3]-[C:4]=[O:5]>>[#16,#8,#7:1]~[*:2]-[CH:3]=[C:4]-[O-:5]"
    ),
    ReactionFromSmarts("[C:1]-[N+:2](=[O:3])-[O-:4]>>[C:1]=[N+:2](-[O-:3])-[O-:4]"),
    ReactionFromSmarts(
        "[O:1]1-[C:2]=[C:3]-[CH2:4]-[C:5]1(=[O:6])>>[O:1]1-[C:2]=[C:3]-[CH:4]=[C:5](-[O-:6])-1"
    ),
    ReactionFromSmarts(
        "[#7H1:1]1~[#7:2]~[#6:3]~[#6:4]~[#6:8]1>>[#7H0-:1]1~[#7:2]~[#6:3]~[#6:4]~[#6:8]1"
    ),
    ReactionFromSmarts("[O:1]=[C:2]-[CH1:3]-[c:5]>>[O-:1]-[C:2]=[CH0:3]-[c:5]"),
    ReactionFromSmarts(
        "[O:1]=[C:2]-[CH1:3]-[C:4]=[O:5]>>[O-:1]-[C:2]=[CH0:3]-[C:4]=[O:5]"
    ),
    ReactionFromSmarts(
        "[N:1]#[C:2]-[C:3](-[C:4]#[N:5])=[C:6]-[CH2:7]>>[N-:1]=[C:2]=[C:3](-[C:4]#[N:5])-[C:6]=[CH1:7]"
    ),
    ReactionFromSmarts("[SH1:1]>>[SH0-:1]"),
    ReactionFromSmarts(
        "[N:1]1-[N:2]-[C:3](=[O:4])-[C:5]-[C:6]=1>>[N:1]1-[N:2]-[C:3](-[O-:4])=[C:5]-[C:6]=1"
    ),
    ReactionFromSmarts("[nH1:1][n:2][n:3]>>[nH0-:1][n:2][n:3]"),
    ReactionFromSmarts("[*:1]=[N:2]-[OH:3]>>[*:1]=[N:2]-[OH0-:3]"),
    ReactionFromSmarts("[C:1]=[C:2]-[CH2:3]-[Sn:4]>>[C:1]=[C:2]-[CH2-1:3].[Sn:4]"),
    ReactionFromSmarts("[N:1]#[CH0:2]-[Si:3]>>[N:1]#[CH0-1:2].[Si:3]"),
    ReactionFromSmarts("[N:1]#[C:2]-[C:3]-[c:4]>>[N-1:1]=[C:2]=[C:3]-[c:4]"),
    ReactionFromSmarts("[O:1]=[*:2]-[N:3]-[NH2:4]>>[O:1]=[*:2]-[N:3]-[NH1-1:4]"),
    ReactionFromSmarts("[a:1]-[O:2]-[Si:3]>>[a:1]-[O-1:2].[Si:3]"),
)
ligand_templates = (
    Chem.MolFromSmarts("[O:1]=CC[N+]([O-:2])CC[N+]([O-:3])CC=[O:4]"),
    Chem.MolFromSmarts("[O:1]=CC[N+]([O-:2])CCC[N+]([O-:3])CC=[O:4]"),
    Chem.MolFromSmarts("[O:1]=CC[N+]([O-:2])CCCC[N+]([O-:3])CC=[O:4]"),
)

metal_coordination_dict = {
    "Mg": [6],
    "Ca": [6],
    "Fe": [6],
    "Co": [6],
    "Ni": [6],
    "Cu": [6],
    "Zn": [6],
    "Sc": [6],
    "In": [6],
    "Tm": [7, 8],
    "Lu": [7, 8],
    "Yb": [7, 8],
    "La": [8],
    "Er": [8],
    "Y": [8],
    "Gd": [8],
    "Ba": [8],
    "Nd": [8],
    "Ce": [8],
    "Pr": [8],
    "Sm": [8],
    "Eu": [8],
    "Gd": [8],
    "Tb": [8],
    "Dy": [8],
    "Ho": [8],
}


def build_catalyst_complex(metal_cation: str, ligand_smiles: str):
    metal_cation_mol = Chem.MolFromSmiles(metal_cation)
    assert metal_cation_mol is not None, f"Invalid metal cation SMILES: {metal_cation}"
    assert (
        metal_cation_mol.GetNumAtoms() == 1
    ), f"Metal cation should have only one atom, but got {metal_cation_mol.GetNumAtoms()}"
    assert (
        metal_cation_mol.GetAtomWithIdx(0).GetSymbol() in metal_coordination_dict
    ), f"Invalid metal cation: {metal_cation_mol.GetAtomWithIdx(0).GetSymbol()}, should be one of {metal_coordination_dict.keys()}"
    assert (
        metal_cation_mol.GetAtomWithIdx(0).GetFormalCharge() >= 0
    ), f"Metal cation should have non-negative formal charge, but got {metal_cation_mol.GetAtomWithIdx(0).GetFormalCharge()}"

    ligand_mol = Chem.MolFromSmiles(ligand_smiles)
    assert ligand_mol is not None, f"Invalid ligand SMILES: {ligand_smiles}"
    for ligand_pattern in ligand_templates:
        if ligand_mol.HasSubstructMatch(ligand_pattern):
            break
    else:
        raise ValueError(f"Invalid ligand: {ligand_smiles}")

    coordinate_number = metal_coordination_dict[
        metal_cation_mol.GetAtomWithIdx(0).GetSymbol()
    ][0]
    waters = [Chem.MolFromSmiles(f"O") for _ in range(coordinate_number - 4)]

    coordinate_index = ligand_mol.GetSubstructMatch(ligand_pattern)

    total_mol = metal_cation_mol
    for sub_mol in waters + [ligand_mol]:
        total_mol = Chem.CombineMols(total_mol, sub_mol)

    total_mol = Chem.RWMol(total_mol)
    for i in range(coordinate_number - 4):
        total_mol.AddBond(i + 1, 0, Chem.BondType.DATIVE)
    for i in [0, 4, -4, -1]:
        total_mol.AddBond(
            coordinate_index[i] + coordinate_number - 3, 0, Chem.BondType.DATIVE
        )
    Chem.SanitizeMol(total_mol)
    return Chem.MolToSmiles(total_mol)


def build_no_extra_catalyst_complex(metal_cation: str, ligand_smiles: str):
    metal_cation_mol = Chem.MolFromSmiles(metal_cation)
    assert metal_cation_mol is not None, f"Invalid metal cation SMILES: {metal_cation}"
    assert (
        metal_cation_mol.GetNumAtoms() == 1
    ), f"Metal cation should have only one atom, but got {metal_cation_mol.GetNumAtoms()}"
    assert (
        metal_cation_mol.GetAtomWithIdx(0).GetSymbol() in metal_coordination_dict
    ), f"Invalid metal cation: {metal_cation_mol.GetAtomWithIdx(0).GetSymbol()}, should be one of {metal_coordination_dict.keys()}"
    assert (
        metal_cation_mol.GetAtomWithIdx(0).GetFormalCharge() >= 0
    ), f"Metal cation should have non-negative formal charge, but got {metal_cation_mol.GetAtomWithIdx(0).GetFormalCharge()}"

    ligand_mol = Chem.MolFromSmiles(ligand_smiles)
    assert ligand_mol is not None, f"Invalid ligand SMILES: {ligand_smiles}"
    for ligand_pattern in ligand_templates:
        if ligand_mol.HasSubstructMatch(ligand_pattern):
            break
    else:
        raise ValueError(f"Invalid ligand: {ligand_smiles}")

    # coordinate_number = metal_coordination_dict[
    #     metal_cation_mol.GetAtomWithIdx(0).GetSymbol()
    # ][0]
    # waters = [Chem.MolFromSmiles(f"O") for _ in range(coordinate_number - 4)]

    coordinate_index = ligand_mol.GetSubstructMatch(ligand_pattern)

    total_mol = metal_cation_mol
    total_mol = Chem.CombineMols(total_mol, ligand_mol)
    # for sub_mol in waters + [ligand_mol]:
    #     total_mol = Chem.CombineMols(total_mol, sub_mol)

    total_mol = Chem.RWMol(total_mol)
    # for i in range(coordinate_number - 4):
    #     total_mol.AddBond(i + 1, 0, Chem.BondType.DATIVE)
    for i in [0, 4, -4, -1]:
        total_mol.AddBond(coordinate_index[i] + 1, 0, Chem.BondType.DATIVE)
    Chem.SanitizeMol(total_mol)
    return Chem.MolToSmiles(total_mol)


def transform_active_state(raw_donor: str) -> str:
    smiles = raw_donor
    mol = Chem.MolFromSmiles(smiles)
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == -1:
        return smiles
    assert mol is not None, f"Invalid donor SMILES: {raw_donor}"
    for template in donor_templates:
        if mol.HasSubstructMatch(template.GetReactants()[0]):
            new_smiles = Chem.MolToSmiles(
                template.RunReactants((mol,))[0][0], kekuleSmiles=True
            )
            break
    else:
        raise ValueError(f"Invalid donor: {raw_donor}")
    return new_smiles


pesudo_templates = {
    ReactionFromSmarts("[*H1,*h1:1]>>[*H0-1,*h0-1:1]"),
    ReactionFromSmarts("[*H2,*h2:1]>>[*H1-1,*h1-1:1]"),
    ReactionFromSmarts("[*H3,*h3:1]>>[*H2-1,*h2-1:1]"),
}


def transform_pesudo_active_state(raw_donor: str) -> str:
    smiles = raw_donor
    mol = Chem.MolFromSmiles(smiles)
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == -1:
        return smiles
    assert mol is not None, f"Invalid donor SMILES: {raw_donor}"
    for template in pesudo_templates:
        if mol.HasSubstructMatch(template.GetReactants()[0]):
            new_smiles = Chem.MolToSmiles(
                template.RunReactants((mol,))[0][0], kekuleSmiles=True
            )
            break
    else:
        raise ValueError(f"Invalid donor: {raw_donor}")
    return new_smiles


def check_atom_mapping_valid(rxn_smiles):
    rxn = ReactionFromSmarts(rxn_smiles)

    def get_mapnums(mols):
        return set(
            atom.GetAtomMapNum()
            for mol in mols
            for atom in mol.GetAtoms()
            if atom.GetAtomMapNum() > 0
        )

    reactant_mapnums = get_mapnums(rxn.GetReactants())
    product_mapnums = get_mapnums(rxn.GetProducts())
    return len(reactant_mapnums) > 0 and reactant_mapnums == product_mapnums


def build_rxn_smiles(
    acceptor: str,
    donor: str,
    product: str,
    salt: str | None = None,
    ligand: str | None = None,
    solvent: str | None = None,
    additive: str | None = None,
    catalyst: str | None = None,
) -> str:
    """
    Build reaction SMILES string.
    """
    pure_rxn = f"{acceptor}.{donor}>>{product}"
    mapped_rxn = map_rxn(pure_rxn)
    conditions = [
        c
        for c in [salt, ligand, solvent, additive, catalyst]
        if (c is not None) and (c != "")
    ]
    if len(conditions) >= 1:
        condition = ".".join(conditions)
        reac, prod = mapped_rxn.split(">>")
        return f"{reac}>{condition}>{prod}"
    else:
        return mapped_rxn


@lru_cache(maxsize=1024)
def map_rxn(pure_rxn: str) -> str:
    if not check_atom_mapping_valid(pure_rxn):
        script = f"from localmapper import localmapper; mapper = localmapper(); print(mapper.get_atom_map('{pure_rxn}'), end='')"
        mapped_rxn = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "localmapper",
                "python",
                "-c",
                script,
            ],  # the environment name should be 'localmapper'
            capture_output=True,
            text=True,
        ).stdout
        mapped_rxn = mapped_rxn.strip().splitlines()[-1]
        if mapped_rxn == "Invalid mol found":
            return pure_rxn
    else:
        mapped_rxn = pure_rxn
    return mapped_rxn


def build_rxn_active_smiles(
    acceptor: str,
    donor: str,
    product: str,
    salt: str | None = None,
    ligand: str | None = None,
    solvent: str | None = None,
    additive: str | None = None,
    catalyst: str | None = None,
) -> str:
    active_donor = transform_active_state(donor)
    return build_rxn_smiles(
        acceptor, active_donor, product, salt, ligand, solvent, additive, catalyst
    )


def build_rxn_pesudo_active_smiles(
    acceptor: str,
    donor: str,
    product: str,
    salt: str | None = None,
    ligand: str | None = None,
    solvent: str | None = None,
    additive: str | None = None,
    catalyst: str | None = None,
) -> str:
    active_donor = transform_pesudo_active_state(donor)
    return build_rxn_smiles(
        acceptor, active_donor, product, salt, ligand, solvent, additive, catalyst
    )


def build_dative_rxn_smiles(
    acceptor: str,
    donor: str,
    product: str,
    salt: str | None = None,
    ligand: str | None = None,
    solvent: str | None = None,
    additive: str | None = None,
    catalyst: str | None = None,
) -> str:
    if (salt is not None and salt != "") and (ligand is not None and ligand != ""):
        salt_mol = Chem.MolFromSmiles(salt)
        assert salt_mol is not None, f"Invalid salt SMILES: {salt}"
        for atom in salt_mol.GetAtoms():
            if atom.GetSymbol() in metal_coordination_dict:
                break
        else:
            raise ValueError("No legal metal atom found in the salt")
        metal_cation = atom.GetSmarts()
        catalyst = build_catalyst_complex(metal_cation, ligand)
        return build_rxn_smiles(
            acceptor,
            donor,
            product,
            solvent=solvent,
            additive=additive,
            catalyst=catalyst,
        )
    else:
        return build_rxn_smiles(
            acceptor, donor, product, salt, ligand, solvent, additive, catalyst
        )


def build_no_extra_dative_rxn_smiles(
    acceptor: str,
    donor: str,
    product: str,
    salt: str | None = None,
    ligand: str | None = None,
    solvent: str | None = None,
    additive: str | None = None,
    catalyst: str | None = None,
) -> str:
    if (salt is not None and salt != "") and (ligand is not None and ligand != ""):
        salt_mol = Chem.MolFromSmiles(salt)
        assert salt_mol is not None, f"Invalid salt SMILES: {salt}"
        for atom in salt_mol.GetAtoms():
            if atom.GetSymbol() in metal_coordination_dict:
                break
        else:
            raise ValueError("No legal metal atom found in the salt")
        metal_cation = atom.GetSmarts()
        catalyst = build_no_extra_catalyst_complex(metal_cation, ligand)
        return build_rxn_smiles(
            acceptor,
            donor,
            product,
            solvent=solvent,
            additive=additive,
            catalyst=catalyst,
        )
    else:
        return build_rxn_smiles(
            acceptor, donor, product, salt, ligand, solvent, additive, catalyst
        )


def build_dative_rxn_active_smiles(
    acceptor: str,
    donor: str,
    product: str,
    salt: str | None = None,
    ligand: str | None = None,
    solvent: str | None = None,
    additive: str | None = None,
    catalyst: str | None = None,
) -> str:
    active_donor = transform_active_state(donor)
    return build_dative_rxn_smiles(
        acceptor, active_donor, product, salt, ligand, solvent, additive, catalyst
    )


def build_no_extra_dative_rxn_pesudo_active_smiles(
    acceptor: str,
    donor: str,
    product: str,
    salt: str | None = None,
    ligand: str | None = None,
    solvent: str | None = None,
    additive: str | None = None,
    catalyst: str | None = None,
) -> str:
    active_donor = transform_pesudo_active_state(donor)
    return build_no_extra_dative_rxn_smiles(
        acceptor, active_donor, product, salt, ligand, solvent, additive, catalyst
    )


def get_reactant_class(smiles: str, mapping: dict) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    for key, value in mapping.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(value["template_smarts"])):
            return key


def get_ticks(mapping: dict) -> list[float]:
    ticks = []
    now_tick = 0
    ticks.append(now_tick)
    for value in list(mapping.values())[:-1]:
        ticks.append(now_tick + value["bandwidth"])
        now_tick += value["bandwidth"]
    return ticks


def get_reactant_indice(smiles: str, mapping: dict) -> float:
    key = get_reactant_class(smiles, mapping)
    ticks = get_ticks(mapping)
    key_idx = list(mapping.keys()).index(key)
    base_mol = Chem.MolFromSmiles(mapping[key]["ref_mol_smiles"])
    mol = Chem.MolFromSmiles(smiles)
    return (
        dice_count_distance(fpgen.fit_transform([base_mol]), fpgen.fit_transform([mol]))
        * mapping[key]["bandwidth"]
        + ticks[key_idx]
    )


def get_donor_subgroup(subclass: str) -> str:
    return {
        "Bidentate-1": "Bidentate",
        "Bidentate-2": "Bidentate",
        "Monodentate-C-1": "Monodentate-C",
        "Monodentate-C-2": "Monodentate-C",
        "Monodentate-C-3": "Monodentate-C",
        "Monodentate-C-cyclic-1": "Monodentate-C-cyclic",
        "Monodentate-C-cyclic-2": "Monodentate-C-cyclic",
        "Monodentate-C-cyclic-3": "Monodentate-C-cyclic",
        "Monodentate-C-cyclic-4": "Monodentate-C-cyclic",
        "Monodentate-C-cyclic-5": "Monodentate-C-cyclic",
        "Monodentate-C-cyclic-6": "Monodentate-C-cyclic",
        "Monodentate-C-cyclic-7": "Monodentate-C-cyclic",
        "Monodentate-N-1": "Monodentate-X",
        "Monodentate-O-1": "Monodentate-X",
        "Monodentate-S-1": "Monodentate-X",
        "Others": "Others",
    }[subclass]


def get_acceptor_subgroup(subclass: str) -> str:
    return {
        "Bidentate-1": "Bidentate",
        "Bidentate-2": "Bidentate",
        "Bidentate-3": "Bidentate",
        "Bidentate-4": "Bidentate",
        "Monodentate-1,2-disub-1": "Monodentate",
        "Monodentate-1,1-disub-1": "Monodentate",
        "Monodentate-1,2-disub-2": "Monodentate",
        "Monodentate-1,1-disub-2": "Monodentate",
        "Monodentate-alkyne-1": "Monodentate",
        "Monodentate-cyclic-2": "Monodentate",
        "Monodentate-cyclic-1": "Monodentate",
        "Others": "Others",
    }[subclass]

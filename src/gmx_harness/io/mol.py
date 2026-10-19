"""MDL .mol files <-> GroFile, for fixing a monomer conformation in a molecule editor.

Needs RDKit (``pip install "gmx-harness[rdkit]"``); it is imported only when these are called.

    save_mol(GroFile.from_gro_file("MOL.gro"), "MOL.mol")          # edit MOL.mol by hand ...
    fixed = gro_from_mol("MOL_fixed.mol", GroFile.from_gro_file("MOL.gro"))
    fixed.save_gro("MOL_fixed.gro")

The .gro keeps the atom and residue names of the template; only the conformation comes from
the .mol, whose atoms must be in the template's order (checked element by element).
"""

import copy
import os
from typing import Any

import numpy as np

from .gro import GroFile

_ANGSTROM_TO_NM = 0.1


def _chem() -> Any:
    try:
        from rdkit import Chem
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('RDKit is needed for .mol files: pip install "gmx-harness[rdkit]"') from e
    return Chem


def save_mol(gro: GroFile, path: str | os.PathLike[str], *, charge: int = 0) -> None:
    """
    Write ``gro`` as an MDL .mol file. Bonds and bond orders are perceived from the coordinates by
    RDKit (``rdDetermineBonds``, total ``charge``); elements are the first letter of each atom name
    (``GroAtom.atom_symbol``), as for ``save_xyz``.
    """
    chem = _chem()
    from rdkit.Chem import rdDetermineBonds

    mol = chem.MolFromXYZBlock("\n".join(gro.generate_xyz_text()) + "\n")
    if mol is None:
        raise ValueError("RDKit could not read the structure as xyz")
    rdDetermineBonds.DetermineConnectivity(mol)
    rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
    chem.MolToMolFile(mol, os.fspath(path))


def gro_from_mol(path: str | os.PathLike[str], template: GroFile) -> GroFile:
    """
    A copy of ``template`` (names, residues, box) with the conformation of the .mol file at ``path``.
    The .mol must have the template's atoms in the same order (hydrogens included); a different
    count or an element that does not match the atom name raises ValueError.
    """
    chem = _chem()
    mol = chem.MolFromMolFile(os.fspath(path), removeHs=False, strictParsing=False)
    if mol is None:
        raise ValueError(f"RDKit could not read {path}")
    if mol.GetNumAtoms() != len(template):
        raise ValueError(f"{path}: {mol.GetNumAtoms()} atoms, template has {len(template)}")
    for i, (atom, ga) in enumerate(zip(mol.GetAtoms(), template.atoms)):
        sym = atom.GetSymbol().upper()
        if not ga.atom_name.upper().startswith(sym):
            raise ValueError(f"{path}: atom {i + 1} is {atom.GetSymbol()} but the template's is {ga.atom_name}; "
                             "the atom order must not change")
    xyz = np.array(mol.GetConformer().GetPositions(), dtype=float) * _ANGSTROM_TO_NM
    out = copy.deepcopy(template)
    for ga, p in zip(out.atoms, xyz):
        ga.coordinate = p
    return out

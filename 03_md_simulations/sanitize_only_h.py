# 03_md_simulations/sanitize_only_h.py
# Strip all H + heterogens from PDB text, normalize altLocs, then rebuild missing atoms
# and add fresh hydrogens at pH 7.4 using PDBFixer. Output is OpenMM-friendly.

import sys, io

from pdbfixer import PDBFixer
from openmm.app import PDBFile


WATER_NAMES = {"HOH", "WAT", "DOD", "H2O"}


def _is_atom_or_het(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


def _is_het(line: str) -> bool:
    return line.startswith("HETATM")


def _resname(line: str) -> str:
    # PDB residue name columns 18-20 (0-based 17:20)
    return line[17:20].strip() if len(line) >= 20 else ""


def _element(line: str) -> str:
    # element columns 77-78 (0-based 76:78) if present
    if len(line) >= 78:
        return line[76:78].strip().upper()
    # fallback: infer from atom name (cols 13-16)
    name = line[12:16].strip() if len(line) >= 16 else ""
    return name[0].upper() if name else ""


def _is_hydrogen(line: str) -> bool:
    if not _is_atom_or_het(line):
        return False
    elem = _element(line)
    if elem == "H":
        return True
    name = line[12:16].strip() if len(line) >= 16 else ""
    return name.startswith("H")


def _normalize_altloc(line: str) -> str:
    # altLoc is column 17 (0-based index 16). Keep ' ' or 'A', blank others.
    if not _is_atom_or_het(line) or len(line) < 17:
        return line
    alt = line[16]
    if alt in (" ", "A"):
        return line
    # replace with blank
    return line[:16] + " " + line[17:]


def strip_h_and_hets_to_stream(pdb_path: str) -> io.StringIO:
    """
    Return an in-memory PDB text with:
      - all H atoms removed
      - all heterogens removed (waters/ligands)
      - only ATOM records kept (no CONECT/REMARK)
      - altLocs normalized (keep ' ' or 'A')
    """
    out = io.StringIO()
    with open(pdb_path, "r") as f:
        for line in f:
            if not _is_atom_or_het(line):
                continue

            # Normalize altLoc before checks
            line = _normalize_altloc(line)

            # Drop heterogens entirely (includes waters)
            if _is_het(line):
                continue
            if _resname(line) in WATER_NAMES:
                continue

            # Drop hydrogens
            if _is_hydrogen(line):
                continue

            # Keep heavy-atom ATOM records
            out.write(line)

    # Ensure a TER/END so PDBFixer is happy
    out.write("TER\nEND\n")
    out.seek(0)
    return out


def main(in_pdb: str, out_pdb: str, ph: float = 7.4) -> None:
    # 1) Strip H + HETATM (waters/ligands) from raw text
    cleaned_stream = strip_h_and_hets_to_stream(in_pdb)

    # 2) Load into PDBFixer and add missing atoms/hydrogens
    fixer = PDBFixer(pdbfile=cleaned_stream)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=float(ph))

    # 3) Write out an OpenMM-friendly PDB
    with open(out_pdb, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sanitize_only_h.py IN.pdb OUT.pdb [pH]")
        sys.exit(1)
    in_pdb = sys.argv[1]
    out_pdb = sys.argv[2]
    ph = float(sys.argv[3]) if len(sys.argv) > 3 else 7.4
    main(in_pdb, out_pdb, ph)

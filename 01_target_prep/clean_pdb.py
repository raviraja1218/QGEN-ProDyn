from Bio.PDB import PDBParser, PDBIO

# Files to clean
files = ["6OIM.pdb", "6VJ0.pdb", "7RPZ.pdb"]

parser = PDBParser()
io = PDBIO()

for f in files:
    structure = parser.get_structure("structure", f)
    # Remove heteroatoms (water/ligands)
    for model in structure:
        for chain in model:
            for residue in list(chain):
                if residue.id[0] != " ":
                    chain.detach_child(residue.id)
    io.set_structure(structure)
    io.save(f.replace(".pdb", "_clean.pdb"))

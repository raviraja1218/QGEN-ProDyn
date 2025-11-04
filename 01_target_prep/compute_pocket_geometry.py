import json, sys
import numpy as np
from Bio.PDB import PDBParser

# Use one hydrogen-added structure as geometry reference
pdb_file = "6OIM_clean_H.pdb"  # you can switch to 7RPZ_clean_H.pdb if preferred

start_res, end_res = 60, 74

parser = PDBParser(QUIET=True)
structure = parser.get_structure("kras", pdb_file)

coords = []
for model in structure:
    for chain in model:  # assumes first chain is the KRAS chain; adjust if needed
        for res in chain:
            if res.id[0] != " ":
                continue
            resseq = res.id[1]
            if start_res <= resseq <= end_res:
                # use CA atom if present, else average heavy atoms
                if "CA" in res:
                    coords.append(res["CA"].coord)
                else:
                    atoms = [a.coord for a in res if a.element != "H"]
                    if atoms:
                        coords.append(np.mean(np.array(atoms), axis=0))
    break  # first model only

coords = np.array(coords)
if coords.size == 0:
    print("No pocket atoms found — check chain/residue numbering.")
    sys.exit(1)

center = coords.mean(axis=0)
mins = coords.min(axis=0)
maxs = coords.max(axis=0)

# add 4 Å padding around the pocket box for docking
padding = np.array([4.0, 4.0, 4.0])
size = (maxs - mins) + 2 * padding

# Save numpy arrays
np.save("pocket_center.npy", center)
np.save("pocket_size.npy", size)

# Also write a Vina-friendly JSON
grid = {
    "center_x": float(center[0]),
    "center_y": float(center[1]),
    "center_z": float(center[2]),
    "size_x": float(size[0]),
    "size_y": float(size[1]),
    "size_z": float(size[2])
}
with open("grid_parameters.json", "w") as f:
    json.dump(grid, f, indent=2)

print("Center:", center)
print("Size:", size)
print("grid_parameters.json written.")


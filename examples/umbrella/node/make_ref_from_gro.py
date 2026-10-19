"""Rewrite PLUMED RMSD reference PDBs with coordinates from an equilibrated .gro.

Usage: python make_ref_from_gro.py <equilibrated.gro> <ref.pdb> [<ref.pdb> ...]

Each ref PDB keeps its atom serials / names / occupancy / beta (written by
umbrella.ref_pdb_text); only x, y, z are replaced by the .gro coordinates of
the same atom serials. The selected atoms are made whole across the periodic box
by chaining minimal images of consecutive atoms (rectangular box), which is how
PLUMED itself reconstructs the group, so the reference is never split by PBC.

Standard library only and Python 3.6 compatible: it runs on the cluster node,
in the grommp.sh of the steered-MD step (umbrella.SteeredMD).
"""

import sys


def read_gro(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[1])
    xyz = [(float(l[20:28]), float(l[28:36]), float(l[36:44])) for l in lines[2:2 + n]]
    box = [float(v) for v in lines[2 + n].split()[:3]]
    return xyz, box


def rewrite(pdb, xyz, box):
    with open(pdb) as f:
        lines = f.read().splitlines()
    out, prev = [], None
    for l in lines:
        if not l.startswith("ATOM"):
            out.append(l)
            continue
        serial = int(l[6:11])
        pos = list(xyz[serial - 1])
        if prev is not None:
            for k in range(3):
                d = pos[k] - prev[k]
                pos[k] -= box[k] * round(d / box[k])
        prev = pos
        out.append("%s%8.3f%8.3f%8.3f%s" % (l[:30], pos[0] * 10, pos[1] * 10, pos[2] * 10, l[54:]))
    with open(pdb, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    xyz, box = read_gro(sys.argv[1])
    for pdb in sys.argv[2:]:
        rewrite(pdb, xyz, box)
        print("rewrote %s from %s" % (pdb, sys.argv[1]))


if __name__ == "__main__":
    main()

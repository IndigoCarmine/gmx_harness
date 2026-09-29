# This script generates intermolecular bond definitions for a system of multiple monomers.
# Each monomer is read from a GRO file, and specified atom pairs are converted to global indices.
# Bonds are created sequentially, with ring closures applied every 'nmols_in_rosette' monomers.
# The resulting bonds are written in GROMACS [ intermolecular_interactions ] format
# (ai aj funct length k), ready to be appended to the topology file.
#
# BEHAVIOUR CHANGE: the ring-closure wrap used to be computed on the 1-based atom
# index directly, which made the closing bond of a rosette come out as atom 0 --
# not a valid GROMACS index -- and, for the second and later rosettes, point at an
# atom belonging to the *previous* rosette. Both are now fixed. Any
# intermolecular_bond.itp generated before this change should be regenerated:
# with natoms=10, nmols_in_rosette=6 the closures move from (51, 0) to (51, 60)
# and from (111, 60) to (111, 120).

def generate_inermolecular_interactions(
    natoms: int,
    nmols: int,
    bonds: list[tuple[int, int]],
    nmols_in_rosette: int = 6,
    bondleng: float = 0.3,  # nm
    k: int = 5000,  # kJ/(mol*nm^2)
    outfile_path: str = "intermolecular_bond.itp",
) -> list[tuple[int, int]]:
    """
    Write an ``[ intermolecular_interactions ]`` bond section (funct 6, no exclusions)
    for ``nmols`` identical monomers of ``natoms`` atoms, grouped in rings of
    ``nmols_in_rosette``. ``bonds`` are 1-based (i, j) pairs from monomer k to the
    next monomer's atoms; the last monomer of each ring wraps to the first.
    Returns the global 1-based atom pairs. Include the file in topo.top (e.g.
    inside ``#ifdef INTER``) and pass ``defines=["INTER"]`` to the MD steps.
    """
    grobalbond = []
    natoms_in_rosette = natoms * nmols_in_rosette
    # generate global bond list
    for i in range(nmols):
        # which rosette molecule i belongs to, and its position within that rosette
        rosette_base = (i // nmols_in_rosette) * natoms_in_rosette
        pos_in_rosette = (i % nmols_in_rosette) * natoms
        for b in bonds:
            # wrap each bond within its own rosette (ring closure), then shift to that rosette.
            # The modulo is taken in 0-based space and shifted back, because GROMACS atom
            # indices are 1-based: wrapping directly on the 1-based value makes the last
            # atom of a rosette come out as 0, which is not a valid index.
            grobalbond.append(
                (
                    rosette_base + ((b[0] - 1 + pos_in_rosette) % natoms_in_rosette) + 1,
                    rosette_base + ((b[1] - 1 + pos_in_rosette) % natoms_in_rosette) + 1,
                )
            )

    #   135   286     1     0.3   5000
    with open(outfile_path, "w", newline="\n") as f:
        f.write("[ intermolecular_interactions ]\n")
        f.write("[ bonds ]\n")
        f.write(";  ai    aj funct   length    k\n")
        for b in grobalbond:
            f.write(f"{b[0]:6d} {b[1]:6d}   6    {bondleng:6.3f}   {k:6.0f}\n")

    return grobalbond

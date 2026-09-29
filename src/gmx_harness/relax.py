"""Relax a structure with a soft-core LJ potential.

For structures built by geometric placement, neighbouring monomers can start
with atom pairs close enough that plain 12-6 LJ gives a practically infinite
force and minimization fails. Here the LJ part of the force field is replaced
by a soft-core form

    4*eps*((sig^6/(r^6+delta^6))^2 - sig^6/(r^6+delta^6))

whose energy stays finite at r = 0 while delta > 0. Minimization is run once
per stage in STAGES, ending at delta = 0 where the expression is exactly the
original LJ, so the final structure is relaxed on the real potential.

The inter-fiber hydrogen bonds that grompp would pull in via ``-DINTER`` are
honoured too, but not through GromacsTopFile: OpenMM has no handler for
GROMACS' [ intermolecular_interactions ], and would silently attribute those
bonds to the last [ moleculetype ] instead. Instead, the atom pairs are read
straight from the #included .itp and pinned to their starting positions with
a stiff CustomExternalForce, which sidesteps the same problem without needing
OpenMM to understand the bond section at all.

Note what that substitution costs: a GROMACS harmonic bond restrains the
*distance* between the pair, so a pair placed too far apart is pulled together
to the bond length. Pinning instead freezes both atoms wherever the geometric
construction put them, at FIXED_ATOM_K stiffness. The H-bonded core therefore
keeps its constructed geometry and only the periphery relaxes. Use
add_intermolecular_bonds() instead if you want the real bonded behaviour.

The final structure is not guaranteed to match what GROMACS would produce, but
it is a good starting point for a subsequent GROMACS minimization.
"""

import os
import re

from openmm import (
    CustomNonbondedForce,
    CustomExternalForce,
    HarmonicBondForce,
    NonbondedForce,
    System,
    VerletIntegrator,
    unit,
)
from openmm.app import GromacsGroFile, GromacsTopFile, Simulation, CutoffPeriodic, Topology
from openmm.unit import Quantity

nanometer = unit.nanometer

# Minimization stages as (soft-core radius / nm, charge scaling). Coulomb has no
# soft core here, so it is switched off while the worst overlaps are still being
# pushed apart and faded back in as delta closes. The last stage must be
# (0, 1) so the structure ends up minimised on the unmodified force field.
STAGES = [(0.3, 0.0), (0.2, 0.0), (0.02, 0.75), (0.0, 1.0)]

if STAGES[-1] != (0.0, 1.0):
    raise ValueError(
        "STAGES must end at (0.0, 1.0) so the final minimization runs on the "
        "unmodified force field"
    )

NONBONDED_CUTOFF = 1.0 * nanometer
MAX_ITERATIONS = 2000
TOLERANCE = 10 * (unit.kilojoule_per_mole / nanometer)

# Spring constant for the positional restraints that stand in for the
# inter-fiber hydrogen bonds. Stiff enough that the pinned atoms do not move
# appreciably under the soft-core forces of the first stage; see the module
# docstring for why pinning is not equivalent to the bonds it replaces.
FIXED_ATOM_K = 1e6 * unit.kilojoule_per_mole / nanometer**2

# GROMACS function type for a plain harmonic bond that generates no exclusions.
# gmx_harness/itp.py writes exactly this type; nothing else is handled here.
_HARMONIC_NO_EXCLUSION = 6

# The name GromacsTopFile gives the CustomNonbondedForce it creates when LJ does
# not fit in a plain NonbondedForce -- for combination rule 1/3 and for
# [ nonbond_params ] alike. See add_softcore_lj.
_GROMACS_LJ_FORCE_NAME = "LennardJonesForce"

# NonbondedForce methods that imply periodicity, mapped onto the
# CustomNonbondedForce equivalent. The reciprocal-space part of Ewald/PME has no
# counterpart on a CustomNonbondedForce, so those collapse to a plain cutoff.
_NONBONDED_METHOD_MAP = {
    NonbondedForce.NoCutoff: CustomNonbondedForce.NoCutoff,
    NonbondedForce.CutoffNonPeriodic: CustomNonbondedForce.CutoffNonPeriodic,
    NonbondedForce.CutoffPeriodic: CustomNonbondedForce.CutoffPeriodic,
    NonbondedForce.Ewald: CustomNonbondedForce.CutoffPeriodic,
    NonbondedForce.PME: CustomNonbondedForce.CutoffPeriodic,
    NonbondedForce.LJPME: CustomNonbondedForce.CutoffPeriodic,
}


def add_softcore_lj(system: System) -> CustomNonbondedForce:
    """Move the LJ part of the NonbondedForce into a soft-core CustomNonbondedForce.

    The 1-4 exception terms stay in the original NonbondedForce; only the plain
    (non-excluded) LJ interactions are softened, which is what blows up on a
    freshly packed bundle. Coulomb also diverges on an overlap, so every charge
    is additionally put behind a ``lambda_q`` parameter offset. Both ``delta``
    and ``lambda_q`` are global parameters, so one System can be re-minimised at
    each stage.

    Only topologies whose LJ lives in the standard NonbondedForce are supported,
    i.e. combination rule 2 with no [ nonbond_params ]. See Raises.

    Raises:
        ValueError: if the system has no NonbondedForce, or if GromacsTopFile
            put the LJ interaction in its own CustomNonbondedForce instead
            (combination rule 1/3, or any [ nonbond_params ] / NBFIX section --
            notably MARTINI). Softening the NonbondedForce would be a silent
            no-op in that case, so it is refused rather than attempted.
    """
    nb = next((f for f in system.getForces() if isinstance(f, NonbondedForce)), None)
    if nb is None:
        raise ValueError("system has no NonbondedForce; there is no LJ term to soften")

    # GromacsTopFile routes LJ into its own CustomNonbondedForce for combination
    # rule 1/3 and whenever [ nonbond_params ] is present. The NonbondedForce
    # then carries charges only, with placeholder sigma/epsilon, so softening it
    # would silently do nothing while the real 12-6 LJ stayed at full strength.
    # Both branches name that force _GROMACS_LJ_FORCE_NAME, so match on the name
    # rather than on the type: a CustomNonbondedForce added for some unrelated
    # reason must not trip this.
    if any(
        isinstance(f, CustomNonbondedForce) and f.getName() == _GROMACS_LJ_FORCE_NAME
        for f in system.getForces()
    ):
        raise ValueError(
            f"the LJ interaction is in a CustomNonbondedForce "
            f"({_GROMACS_LJ_FORCE_NAME!r}), not the NonbondedForce: this topology "
            "uses combination rule 1 or 3, or a [ nonbond_params ] section "
            "(e.g. MARTINI). Softening the NonbondedForce would leave the real LJ "
            "untouched, so it is refused rather than silently doing nothing. Use a "
            "combination-rule-2 topology, or extend this function to soften that "
            "force instead."
        )

    softcore = CustomNonbondedForce(
        "4*epsilon*((sigma^6/(r^6+delta^6))^2-sigma^6/(r^6+delta^6));"
        "sigma=0.5*(sigma1+sigma2);"
        "epsilon=sqrt(epsilon1*epsilon2)"
    )
    softcore.addGlobalParameter("delta", STAGES[0][0])
    softcore.addPerParticleParameter("sigma")
    softcore.addPerParticleParameter("epsilon")
    # Mirror the source force rather than assuming periodicity, so this helper
    # stays correct for systems built with a different nonbonded method.
    source_method = nb.getNonbondedMethod()
    if source_method not in _NONBONDED_METHOD_MAP:
        raise ValueError(
            f"unsupported NonbondedForce method {source_method}; cannot pick a "
            "matching CustomNonbondedForce method for the soft-core term"
        )
    softcore.setNonbondedMethod(_NONBONDED_METHOD_MAP[source_method])
    softcore.setCutoffDistance(nb.getCutoffDistance())
    softcore.setUseSwitchingFunction(False)
    # The soft-core form is not a pure r^-6 tail, so the analytic LJ tail
    # correction does not apply; drop it on both forces to keep them consistent.
    softcore.setUseLongRangeCorrection(False)
    nb.setUseDispersionCorrection(False)

    nb.addGlobalParameter("lambda_q", STAGES[0][1])
    for index in range(nb.getNumParticles()):
        charge, sigma, epsilon = nb.getParticleParameters(index)
        softcore.addParticle([sigma, epsilon])
        nb.setParticleParameters(index, 0.0, sigma, 0.0)
        if charge.value_in_unit(unit.elementary_charge) != 0.0:
            nb.addParticleParameterOffset("lambda_q", index, charge, 0.0, 0.0)

    for index in range(nb.getNumExceptions()):
        p1, p2, chargeprod, sigma, epsilon = nb.getExceptionParameters(index)
        softcore.addExclusion(p1, p2)
        # Exception charges are handled directly rather than via the softened
        # force, so they need the same fade-in as the particle charges.
        if chargeprod.value_in_unit(unit.elementary_charge**2) != 0.0:
            nb.setExceptionParameters(index, p1, p2, 0.0, sigma, epsilon)
            nb.addExceptionParameterOffset("lambda_q", index, chargeprod, 0.0, 0.0)

    system.addForce(softcore)
    return softcore


def find_inter_itp(top_path: str) -> str | None:
    """Return the .itp that ``top_path`` #includes inside its #ifdef INTER block.

    This is the file grompp -DINTER would pull in; None if the .top has no such
    block (the generated single-monomer tops don't).
    """
    with open(top_path, "r", encoding="utf-8") as file:
        text = file.read()

    block = re.search(r"^#ifdef\s+INTER\b(.*?)^#endif", text, re.S | re.M)
    if block is None:
        return None

    include = re.search(r'^#include\s+"([^"]+)"', block.group(1), re.M)
    if include is None:
        return None

    return os.path.join(os.path.dirname(os.path.abspath(top_path)), include.group(1))


def parse_inter_bonds(itp_path: str) -> list[tuple[int, int, float, float]]:
    """Parse the [ bonds ] rows of an [ intermolecular_interactions ] .itp.

    Returns one ``(i, j, length_nm, k)`` tuple per bond, with ``i``/``j``
    converted to 0-based atom indices. This is the single parser for the format;
    both add_intermolecular_bonds() and relax() go through it so the two cannot
    drift apart. The producer is gmx_harness/itp.py.

    Only function type 6 is accepted: that is GROMACS' plain harmonic bond that
    generates no exclusions, which maps exactly onto a HarmonicBondForce and
    leaves the nonbonded setup untouched. Atom indices in this section are
    1-based and global over the whole system, matching OpenMM's ordering.

    Raises:
        ValueError: if a bond row has fewer than 5 fields, carries a function
            type other than 6, holds a non-integer or non-positive atom index,
            or if the file contains no [ bonds ] section. Every message names
            the file and line so the offending row can be found.
    """
    bonds: list[tuple[int, int, float, float]] = []
    category: str | None = None
    seen_bonds_section = False

    with open(itp_path, "r", encoding="utf-8") as file:
        for lineno, raw in enumerate(file, 1):
            line = raw.split(";")[0].strip()
            if not line:
                continue
            # Preprocessor directives (#include, #ifdef, ...) are routine in .itp
            # files and are not bond rows.
            if line.startswith("#"):
                continue
            if line.startswith("["):
                category = line.strip("[] ").lower()
                if category == "bonds":
                    seen_bonds_section = True
                continue
            if category != "bonds":
                continue

            fields = line.split()
            if len(fields) < 5:
                raise ValueError(
                    f"{itp_path}:{lineno}: expected 'ai aj funct length k', got {line!r}"
                )

            try:
                i, j, funct = int(fields[0]), int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"{itp_path}:{lineno}: atom indices and function type must be "
                    f"integers, got {line!r}"
                ) from exc

            if funct != _HARMONIC_NO_EXCLUSION:
                raise ValueError(
                    f"{itp_path}:{lineno}: unsupported intermolecular bond type "
                    f"{funct} for atoms {i}-{j} (only type "
                    f"{_HARMONIC_NO_EXCLUSION} is handled)"
                )

            # GROMACS atom indices are 1-based, so 0 (or negative) is malformed.
            # Catching it here rather than after the -1 conversion keeps the
            # message in the user's coordinate system, and stops a 0 from
            # becoming a negative index that silently wraps to the last atom.
            for field_name, value in (("ai", i), ("aj", j)):
                if value < 1:
                    raise ValueError(
                        f"{itp_path}:{lineno}: {field_name} = {value} is not a valid "
                        "1-based GROMACS atom index (check the ring-closure wrap in "
                        "generate_inermolecular_interactions)"
                    )

            try:
                length, k = float(fields[3]), float(fields[4])
            except ValueError as exc:
                raise ValueError(
                    f"{itp_path}:{lineno}: length and k must be numeric, got {line!r}"
                ) from exc

            bonds.append((i - 1, j - 1, length, k))

    if not seen_bonds_section:
        raise ValueError(f"{itp_path}: no [ bonds ] found under intermolecular interactions")
    if not bonds:
        raise ValueError(f"{itp_path}: [ bonds ] section is empty")

    return bonds


def add_intermolecular_bonds(system: System, itp_path: str) -> int:
    """Add the [ intermolecular_interactions ] bonds of ``itp_path`` to ``system``.

    This reproduces what grompp -DINTER would do: a real harmonic bond that
    restrains the distance between each pair. relax() pins the atoms instead --
    see the module docstring for the trade-off between the two.

    Returns:
        The number of bonds added.

    Raises:
        ValueError: for anything parse_inter_bonds() rejects, or if a bond
            references an atom outside the system.
    """
    force = HarmonicBondForce()
    force.setName("IntermolecularHBonds")

    bonds = parse_inter_bonds(itp_path)
    n_particles = system.getNumParticles()
    for i, j, length, k in bonds:
        for index in (i, j):
            if not 0 <= index < n_particles:
                raise ValueError(
                    f"{itp_path}: bond references atom {index + 1}, but the "
                    f"system has only {n_particles} atoms"
                )
        force.addBond(i, j, length * nanometer, k * unit.kilojoule_per_mole / nanometer**2)

    system.addForce(force)
    return int(force.getNumBonds())


def add_fixed_atoms(
    system: System,
    positions: Quantity,
    fixed_indices: list[int],
    k: Quantity = FIXED_ATOM_K,
) -> int:
    """Add a CustomExternalForce pinning the given atoms to their positions.

    The restraint is harmonic with a very large spring constant by default, so
    the atoms stay effectively in place. Reference coordinates must be passed in:
    a System carries no positions, they only exist once a Context is built.

    Indices are deduplicated, since an atom appearing in two hydrogen bonds
    would otherwise be pinned twice and feel double the spring constant.

    Args:
        system: System to add the restraint force to.
        positions: Reference coordinates for every particle in ``system``.
        fixed_indices: 0-based indices of the atoms to pin.
        k: Spring constant; defaults to FIXED_ATOM_K.

    Returns:
        The number of atoms pinned (after deduplication), or 0 if
        ``fixed_indices`` was empty, in which case no force is added.

    Raises:
        ValueError: if any index falls outside ``0 <= index < numParticles``.
    """
    if not fixed_indices:
        return 0

    fixed_force = CustomExternalForce("0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
    fixed_force.setName("FixedAtoms")
    fixed_force.addPerParticleParameter("x0")
    fixed_force.addPerParticleParameter("y0")
    fixed_force.addPerParticleParameter("z0")
    fixed_force.addGlobalParameter("k", k)

    n_particles = system.getNumParticles()
    coords = positions.value_in_unit(nanometer)
    for index in sorted(set(fixed_indices)):
        # Both bounds matter: a negative index would pass an upper-bound-only
        # check and then silently pin the *last* atom via Python's negative
        # indexing on ``coords``.
        if not 0 <= index < n_particles:
            raise ValueError(
                f"fixed atom index {index} is outside the system "
                f"(0 <= index < {n_particles})"
            )
        x, y, z = coords[index]
        fixed_force.addParticle(index, [x, y, z])

    system.addForce(fixed_force)
    return int(fixed_force.getNumParticles())


def _assert_rectangular_box(box_vectors: Quantity) -> tuple[float, float, float]:
    """Return the box diagonal, rejecting a triclinic cell.

    Raises:
        NotImplementedError: if any off-diagonal component is non-zero.
    """
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = box_vectors.value_in_unit(nanometer)
    off_diagonal = (ay, az, bx, bz, cx, cy)
    if any(abs(v) > 1e-9 for v in off_diagonal):
        raise NotImplementedError(
            "only rectangular boxes are supported, but the box vectors have "
            f"non-zero off-diagonal components {off_diagonal}"
        )
    return ax, by, cz


def write_gro(
    topology: Topology,
    positions: Quantity,
    box_vectors: Quantity,
    out_path: str,
    title: str,
) -> None:
    """Write a GROMACS .gro. OpenMM ships a reader but no writer for this format.

    gmx_harness/io/gro.py also writes this format, from its own GroFile/GroAtom model.
    This function is not built on it because it works from an OpenMM Topology
    plus Quantity positions, which would need a full conversion to GroAtom
    objects first. The record layout below must stay in step with
    GroAtom.__str__ there.

    Raises:
        NotImplementedError: if ``box_vectors`` is triclinic. The 3-value box
            line written here can only express a rectangular cell, and silently
            dropping the off-diagonal terms would change the system's
            periodicity between this relaxation and the GROMACS run that follows.
        ValueError: if a residue carries a non-numeric id.
    """
    coords = positions.value_in_unit(nanometer)
    atoms = list(topology.atoms())

    # (i5,2a5,i5,3f8.3) -- residue number, residue name (left-justified), atom
    # name (right-justified), atom number, then the coordinates. Numbers wrap at
    # 100000 because the fields are only 5 wide; that is GROMACS' own behaviour.
    ax, by, cz = _assert_rectangular_box(box_vectors)

    with open(out_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(f"{title}\n")
        file.write(f"{len(atoms):5d}\n")
        for atom in atoms:
            x, y, z = coords[atom.index]
            residue = atom.residue
            try:
                residue_number = int(residue.id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"residue {residue.name!r} has non-numeric id {residue.id!r}; "
                    "write_gro needs an integer residue number"
                ) from exc
            file.write(
                "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
                % (
                    (residue_number % 100000),
                    residue.name[:5],
                    atom.name[:5],
                    ((atom.index + 1) % 100000),
                    x,
                    y,
                    z,
                )
            )
        file.write("%10.5f%10.5f%10.5f\n" % (ax, by, cz))


def relax(
    gro_path: str,
    top_path: str,
    out_dir: str,
    include_dir: str | None = None,
    fixed_atom_k: Quantity = FIXED_ATOM_K,
) -> str:
    """Relax ``gro_path``/``top_path`` and write ``<stem>_relaxed.gro`` into ``out_dir``.

    ``include_dir`` is forwarded to GromacsTopFile for resolving ``#include``s
    that live outside the .top file's own directory (e.g. a shared
    force-field directory); the .top file's own directory is always searched
    in addition, so pass None when every ``#include`` sits next to it.

    ``fixed_atom_k`` tunes the stiffness of the positional restraints that stand
    in for the inter-fiber hydrogen bonds.

    Returns:
        The path of the written .gro file.

    Raises:
        NotADirectoryError: if ``out_dir`` does not exist. Checked up front so a
            typo fails immediately rather than after the whole minimization.
        ValueError: for an unsupported topology (see add_softcore_lj) or a
            malformed inter-fiber .itp (see parse_inter_bonds).
    """
    if not os.path.isdir(out_dir):
        raise NotADirectoryError(f"out_dir does not exist: {out_dir}")

    gro = GromacsGroFile(gro_path)
    box = gro.getPeriodicBoxVectors()
    # Minimization keeps the box fixed, so a cell write_gro cannot express at the
    # end is already unusable now. Fail here rather than after the full run.
    _assert_rectangular_box(box)
    top = GromacsTopFile(
        top_path,
        periodicBoxVectors=box,
        includeDir=include_dir,
    )
    system = top.createSystem(
        nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=NONBONDED_CUTOFF,
    )
    add_softcore_lj(system)

    # Pin the C=O/NH atoms listed in the #ifdef INTER itp to their initial
    # positions instead of restraining them with a HarmonicBondForce: see the
    # module docstring for why the bonds can't go through GromacsTopFile, and
    # what the substitution costs.
    inter_itp = find_inter_itp(top_path)
    if inter_itp is None:
        print("    no #ifdef INTER block; relaxing without inter-fiber restraints")
    else:
        fixed_indices: list[int] = []
        for i, j, _length, _k in parse_inter_bonds(inter_itp):
            fixed_indices.append(i)
            fixed_indices.append(j)

        count = add_fixed_atoms(system, gro.positions, fixed_indices, k=fixed_atom_k)
        print(f"    fixed {count} atoms from {os.path.basename(inter_itp)}")

    simulation = Simulation(top.topology, system, VerletIntegrator(0.001 * unit.picosecond))
    simulation.context.setPositions(gro.positions)

    for delta, lambda_q in STAGES:
        simulation.context.setParameter("delta", delta)
        simulation.context.setParameter("lambda_q", lambda_q)
        before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        simulation.minimizeEnergy(tolerance=TOLERANCE, maxIterations=MAX_ITERATIONS)
        after = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(
            f"    delta={delta:4.2f} nm q={lambda_q:4.2f}: {before} -> {after}",
            flush=True,
        )

    state = simulation.context.getState(getPositions=True)
    stem = os.path.splitext(os.path.basename(gro_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_relaxed.gro")
    write_gro(
        top.topology,
        state.getPositions(),
        state.getPeriodicBoxVectors(),
        out_path,
        f"{stem} relaxed with soft-core LJ",
    )
    return out_path

"""Design checks: stop a wrong design before anything is computed.

Every check returns a ``Report`` of ``Issue``s with stable codes (``CODES``);
``report.enforce(WAIVE)`` raises ``HarnessCheckError`` unless every error is
waived with a reason, ``WAIVE = {"S004": "box tested in run 123, no self-contact"}``.

Layers (all pure Python; nothing here runs GROMACS or PLUMED):

- ``facts``: values handed from one stage script to the next (``record_facts`` / ``expect``)
- ``structure``: gro / top / itp / ndx consistency
- ``plumed``: static PLUMED input checks, template expansion with unused-define detection
- ``plan``: a planned pipeline (``build_plan(..., checks=True)`` runs it)
- ``postrun``: logs and COLVAR after a run

``python -m gmx_harness.checks <dir>`` lists the facts files below ``dir``.
"""

from .facts import check_fresh, expect, facts_path, lookup, read_facts, record_facts
from .plan import check_plan
from .plumed import check_plumed, expand_template, parse_plumed
from .postrun import check_step, check_tree, read_colvar
from .report import CODES, HarnessCheckError, Issue, Report, validate_waivers
from .structure import (
    check_bond_lengths,
    check_bond_pairs,
    check_box,
    check_contacts,
    check_labels,
    check_ndx,
    check_periodic_twist,
    check_topology,
    check_whole_molecules,
    topology_molecules,
)

__all__ = [
    "CODES",
    "HarnessCheckError",
    "Issue",
    "Report",
    "validate_waivers",
    "record_facts",
    "expect",
    "check_fresh",
    "read_facts",
    "lookup",
    "facts_path",
    "check_topology",
    "check_whole_molecules",
    "check_box",
    "check_periodic_twist",
    "check_bond_pairs",
    "check_bond_lengths",
    "check_contacts",
    "check_labels",
    "check_ndx",
    "topology_molecules",
    "check_plumed",
    "parse_plumed",
    "expand_template",
    "check_plan",
    "check_step",
    "check_tree",
    "read_colvar",
]

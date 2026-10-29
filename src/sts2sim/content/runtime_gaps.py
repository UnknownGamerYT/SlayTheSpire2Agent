"""Concrete source-behavior gaps found by the bounded runtime audit.

These are explicit known gaps, separate from cards whose behavior has not yet
been audited. Remove a row only with source-backed runtime regression evidence;
the compact audit inventory records the corresponding fix commits and tests.
"""

from collections.abc import Mapping
from types import MappingProxyType

# All 81 findings in docs/runtime_audit_inventory.json now have recorded
# source-predicate regression evidence. This is not a parity certificate.
KNOWN_CARD_RUNTIME_GAPS: Mapping[str, str] = MappingProxyType({})

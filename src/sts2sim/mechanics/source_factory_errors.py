"""Explicit failures defined by the stable CardFactory source contract."""

from __future__ import annotations


class SourceCardFactoryError(ValueError):
    """The source factory faults instead of generating a partial reward.

    RNG objects retain draws made before the fault. These fields describe
    factory work only; they do not assert how the installed UI recovers.
    """

    source_exception_type = "InvalidOperationException"
    source_member = "CardFactory.CreateForReward"

    def __init__(
        self,
        reason: str,
        *,
        rolled_rarity: str | None = None,
        generated_card_ids: tuple[str, ...] = (),
    ) -> None:
        self.reason = reason
        self.rolled_rarity = rolled_rarity
        self.generated_card_ids = generated_card_ids
        super().__init__(f"Stable source card factory failed: {reason}")

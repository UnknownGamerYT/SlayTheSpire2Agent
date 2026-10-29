"""Royalties queues an ordinary GoldReward; it never grants Gold at victory."""
from __future__ import annotations

from .models import CombatState, EffectEvent


def after_combat(combat: CombatState) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    amount = t._status_amount(combat.player.statuses, "royalties")
    if (amount <= 0 or combat.player.hp <= 0
            or combat.metadata.get("source_royalties_reward_queued")):
        return combat, ()
    pending = tuple(t._source_sequence(combat.metadata.get("pending_gold_rewards")))
    queue = (tuple(t._source_sequence(combat.metadata["source_extra_reward_queue"]))
             if "source_extra_reward_queue" in combat.metadata else tuple(
                 {**t._mapping_from(item), "kind": "gold"} for item in pending
             ))
    component = {"amount": amount, "source_id": "royalties", "was_gold_stolen_back": False}
    return combat.model_copy(update={"metadata": {
        **combat.metadata, "source_royalties_reward_queued": True,
        "pending_gold_rewards": pending + (component,),
        "source_extra_reward_queue": queue + ({**component, "kind": "gold"},),
    }}), (EffectEvent(kind="extra_gold_reward_added", source_id="royalties", amount=amount,
                     metadata={"deferred_until_claim": True}),)

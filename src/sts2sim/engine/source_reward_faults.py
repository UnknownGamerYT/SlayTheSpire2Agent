"""Oracle traces for deterministic reward faults in the pinned game source.

These traces describe the source prefix through its exception. They are not
normal transitions and do not invent an installed-game UI recovery path.
"""
from __future__ import annotations

from typing import Any

from sts2sim.mechanics.source_card_factory import source_noncombat_cards
from sts2sim.mechanics.source_factory_errors import SourceCardFactoryError
from sts2sim.mechanics.source_rng import is_source_rng

from .models import RunState
from .rng import capture_random_state, random_from_state
from .source_reward_reroll import _group_ids, _pool, reward_reroll_contexts


def inspect_reward_reroll_fault(state: RunState, group: str = "primary") -> dict[str, Any]:
    """Inspect a known source fault without mutating or advancing the input run.

    The returned RNG snapshots are oracle data. Do not put this diagnostic in
    policy observations or use the partial reward model as a normal successor.
    """
    reward = state.reward
    if reward is None or "driftwood" not in state.relics:
        raise ValueError("A pending card reward and Driftwood are required")
    context = reward_reroll_contexts(reward).get(group)
    if context is None or context.get("source_fault_reason") != "empty_custom_card_pool":
        raise ValueError("This reward group has no modeled source reroll fault")
    previous = _group_ids(reward, group)
    used = tuple(reward.metadata.get("card_reward_rerolled_groups", ()))
    if not previous or group in used:
        raise ValueError("The source card reward can no longer be rerolled")
    rng = random_from_state(state.rng)
    if not is_source_rng(rng):
        raise ValueError("A stable-source RNG snapshot is required for this oracle trace")

    # CardReward.Reroll mutates these fields before Populate calls CardFactory.
    metadata = {**reward.metadata, "card_reward_rerolled_groups": (*used, group)}
    if group == "primary":
        partial_reward = reward.model_copy(update={"card_options": (), "metadata": metadata})
    else:
        index = int(group.split(":", 1)[1])
        groups = list(reward.card_option_groups)
        groups[index] = ()
        partial_reward = reward.model_copy(update={
            "card_option_groups": tuple(groups), "metadata": metadata,
        })
    try:
        source_noncombat_cards(
            rng, _pool(state, context), int(context.get("count", len(previous))),
            uniform=bool(context.get("uniform")),
            no_upgrade_roll=bool(context.get("no_upgrade_roll", True)),
            act=state.act, ascension=state.ascension,
        )
    except SourceCardFactoryError as exc:
        rng_after = capture_random_state(rng)
        return {
            "classification": "modeled_source_fault",
            "normal_transition": False,
            "group": group,
            "original_state_unchanged": True,
            "source_exception": {
                "type": exc.source_exception_type,
                "member": exc.source_member,
                "reason": exc.reason,
                "rolled_rarity": exc.rolled_rarity,
            },
            "source_prefix": (
                "disable_reroll", "append_unpicked_card_history", "mark_rerolled",
                "clear_cards", "modify_creation_options", "roll_base_rarity",
                "throw_no_valid_rarity",
            ),
            "history_append": tuple({"card_id": identifier, "was_picked": False}
                                    for identifier in previous),
            "partial_reward_model": partial_reward.model_dump(mode="json"),
            "source_card_reward_fields": {
                "can_reroll": False, "has_been_rerolled": True, "cards": (),
            },
            "rng_before": state.rng.model_dump(mode="json"),
            "rng_after": rng_after.model_dump(mode="json"),
            "rarity_offset_changed": False,
            "factory_returned": False,
            "after_generated_invoked": False,
            "reward_taken_hooks_invoked": False,
            "continuation_requires_live_ui_observation": True,
        }
    raise ValueError("The recorded source fault did not occur; no UI continuation was inferred")

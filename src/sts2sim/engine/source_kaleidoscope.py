"""Kaleidoscope's two manual rewards, each from three distinct character pools.

Source model/file hashes and the empty-custom-pool reroll boundary are recorded
in tests/fixtures/source_kaleidoscope.json. Game-source code is not bundled.
"""
from __future__ import annotations

from random import Random
from typing import Any

from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind
from sts2sim.mechanics.source_ancients import source_unlocked_characters
from sts2sim.mechanics.source_card_factory import source_noncombat_cards

from .models import EffectEvent, RewardState, RunPhase, RunState
from .rng import capture_random_state, random_stream, stable_shuffle
from .source_card_reward_modifiers import modify_generated_cards
from .source_reward_reroll import card_reroll_context


def pickup(state: RunState, rng: Random) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    other_pools = tuple(pool for pool in source_unlocked_characters(state.flags)
                        if pool != state.character_id.lower())
    reward = state.reward or RewardState(
        reward_id=f"relic:kaleidoscope:{state.act}:{state.floor}", source="ancient",
    )
    primary = reward.card_options
    groups = list(reward.card_option_groups)
    metadata = dict(reward.metadata)
    primary_upgraded = tuple(metadata.get("card_option_upgraded", ()))
    group_upgrades = list(metadata.get("card_option_group_upgraded", ()))
    group_rarities = list(metadata.get("card_group_rarities", ()))
    contexts = dict(t._mapping_from(metadata.get("card_reroll_contexts")))
    option_modifiers = dict(t._mapping_from(metadata.get("card_reward_option_modifiers")))
    applied = set(t._string_items(metadata.get("source_modified_card_reward_groups")))
    modified_relics = dict(t._mapping_from(metadata.get("source_card_reward_modified_relics")))
    unsupported = dict(t._mapping_from(metadata.get("source_unsupported_card_reward_rerolls")))
    pool_groups: list[tuple[str, ...]] = []
    events: list[EffectEvent] = []
    for group_number in range(2):
        selected_pools = tuple(stable_shuffle(
            other_pools, random_stream(rng, "niche"), key=lambda pool: pool.upper(),
        )[:3])
        identifiers: list[str] = []
        rarities: list[str] = []
        upgrades: list[bool] = []
        modifiers: dict[str, Any] = {}
        for pool_id in selected_pools:
            marker = EventFlowMarker(kind=EventFlowMarkerKind.CARD_REWARD, qualifier=pool_id)
            ids, rarity, upgraded = source_noncombat_cards(
                rng, tuple(card for card in t._event_flow_card_pool(state, marker)
                           if card.color.lower() == pool_id), 1,
                no_upgrade_roll=False, act=state.act, ascension=state.ascension,
            )
            if not ids:
                events.append(EffectEvent(
                    kind="relic_card_reward_blocked", source_id="kaleidoscope",
                    metadata={"reason": "missing_selected_character_pool", "pool": pool_id},
                ))
                continue
            slot = len(identifiers)
            state, upgraded, card_modifiers, changed = modify_generated_cards(
                state, rng, ids, upgraded, is_card_reward=False,
                context={"group_key": f"kaleidoscope:{group_number}:{pool_id}",
                         "no_card_pool_modifications": True},
            )
            events.extend(changed)
            identifiers.extend(ids)
            rarities.extend(rarity)
            upgrades.extend(upgraded)
            if "0" in card_modifiers:
                modifiers[str(slot)] = card_modifiers["0"]
        if not identifiers:
            continue
        # The manually supplied CardReward calls the hooks again with
        # IsCardReward, even though NoCardModelModifications is also set.
        state, upgraded_group, modifiers, changed = modify_generated_cards(
            state, rng, identifiers, upgrades, is_card_reward=True,
            context={"group_key": f"kaleidoscope:manual:{group_number}",
                     "no_card_pool_modifications": True, "no_card_model_modifications": True},
            current_modifiers=modifiers,
        )
        events.extend(changed)
        if not primary and not reward.card_claimed and not reward.card_skipped:
            key = "primary"
            primary, primary_upgraded = tuple(identifiers), upgraded_group
            metadata["card_rarities"] = tuple(rarities)
        else:
            index = len(groups)
            key = f"group:{index}"
            groups.append(tuple(identifiers))
            while len(group_upgrades) <= index:
                group_upgrades.append(())
            while len(group_rarities) <= index:
                group_rarities.append(())
            group_upgrades[index], group_rarities[index] = upgraded_group, tuple(rarities)
        option_modifiers[key] = modifiers
        applied.add(key)
        modified_relics[key] = tuple(t._normalized_id(relic) for relic in state.relics)
        # Source passes an explicit empty custom pool as RerollOptions. It is
        # not the owning character's pool. Keep the unsupported boundary visible.
        reason = "source_empty_custom_card_pool"
        contexts[key] = card_reroll_context(
            mode="noncombat", pools=(), count=len(identifiers), card_ids=(),
            custom_pool=True, no_upgrade_roll=True, unsupported_reason=reason,
            availability="unsupported",
        )
        unsupported[key] = {"reason": reason, "source": "Kaleidoscope.AfterObtained"}
        pool_groups.append(selected_pools)
        events.append(EffectEvent(
            kind="relic_card_reward_generated", source_id="kaleidoscope",
            target_id=key, amount=len(identifiers), metadata={
                "card_ids": tuple(identifiers), "character_pools": selected_pools,
                "card_rarities": tuple(rarities), "card_upgraded": upgraded_group,
            },
        ))
    metadata.update(
        card_option_upgraded=primary_upgraded, card_option_group_upgraded=tuple(group_upgrades),
        card_group_rarities=tuple(group_rarities), card_reroll_contexts=contexts,
        card_reward_option_modifiers=option_modifiers,
        source_modified_card_reward_groups=tuple(sorted(applied)),
        source_card_reward_modified_relics=modified_relics,
        source_unsupported_card_reward_rerolls=unsupported,
        kaleidoscope_character_pool_groups=tuple(pool_groups),
        card_reward_group_count=bool(primary) + len(groups),
    )
    reward = reward.model_copy(update={
        "card_options": primary, "card_option_groups": tuple(groups), "metadata": metadata,
    })
    return state.model_copy(update={
        "reward": reward, "rng": capture_random_state(rng),
        "phase": (RunPhase.REWARD
                  if state.phase in {RunPhase.ANCIENT, RunPhase.MAP} else state.phase),
    }), tuple(events)

"""Ordered CombatRoom.ExtraRewards descriptors for source single-player runs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import RunState

KEY = 'source_extra_reward_queue'


def initial_extra_rewards(state: RunState) -> tuple[dict[str, Any], ...]:
    """Event parameters exist before later combat/death rewards are appended."""
    from . import transitions as t

    identifier = t._normalized_id(str(state.flags.get('event_fight_id', '')))
    if identifier == 'punch_off':
        return ({'kind': 'relic', 'source_id': identifier},
                {'kind': 'potion', 'source_id': identifier})
    if identifier == 'fake_merchant' or t._is_fake_merchant_combat(state):
        return tuple({'kind': 'relic', 'source_id': 'fake_merchant', 'relic_id': relic}
                     for relic in t._combat_reward_relic_ids(state, default_relic_ids=()))
    if identifier == 'the_lantern_key':
        return ({'kind': 'special_card', 'source_id': identifier, 'card_id': 'lantern_key'},)
    return ()


def append_extra_reward(
    metadata: Mapping[str, Any], descriptor: Mapping[str, Any], *,
    before_fatal_monster_id: str | None = None,
) -> dict[str, Any]:
    """Append a source reward while retaining a queue's creation order.

    Heist's BeforeDeath precedes The Hunt's post-attack Fatal reward. The legacy
    death-script adapter observes the former later, so insert it before that
    same victim's fatal descriptor instead of reversing the source order.
    """
    raw = metadata.get(KEY, ())
    queue = [dict(value) for value in raw if isinstance(value, Mapping)]
    index = len(queue)
    if before_fatal_monster_id is not None:
        index = next((i for i, value in enumerate(queue)
                      if value.get('kind') == 'card'
                      and value.get('defeated_monster_id') == before_fatal_monster_id), index)
    queue.insert(index, dict(descriptor))
    return {**metadata, KEY: tuple(queue)}


def extra_rewards(state: RunState) -> tuple[dict[str, Any], ...]:
    """Read authoritative queues; retain older single-kind save descriptors."""
    from . import transitions as t

    metadata = state.combat.metadata if state.combat is not None else {}
    if KEY in metadata:
        return tuple(dict(value) for value in metadata[KEY] if isinstance(value, Mapping))
    # Existing old snapshots did not retain mixed descriptor order. New source
    # combats initialize KEY before any producer. Legacy metadata fixtures keep
    # their explicit values rather than silently losing rewards during loading.
    result = list(initial_extra_rewards(state))
    result.extend({'kind': 'gold', **dict(value)} for value in
                  t._source_sequence(metadata.get('pending_gold_rewards'))
                  if isinstance(value, Mapping))
    result.extend({'kind': 'card', 'source_id': 'the_hunt', 'encounter': 'inherit'}
                  for _ in range(t._metadata_int(metadata, 'fatal_card_reward_count', 0)))
    return tuple(result)

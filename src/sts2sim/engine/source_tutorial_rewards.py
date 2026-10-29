"""Stable first-run Ironclad reward presets and map-point counting.

RewardsSet.TryGenerateTutorialRewards replaces base generation, before ordinary
ModifyRewards hooks append their rewards. Manual cards, potions and relics do
not consume factory RNG; GoldReward.Populate still draws once.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

from sts2sim.mechanics.combat_rewards import (
    DEFAULT_COMBAT_REWARD_RULES,
    CombatRewardBundle,
    CombatRewardContext,
    _roll_gold,
)
from sts2sim.mechanics.rewards import EncounterType, RewardPityState
from sts2sim.mechanics.source_rng import is_source_rng

from .models import MapNodeState, RoomKind, RunState
from .source_reward_reroll import card_reroll_context

HISTORY = 'source_tutorial_room_history'
MONSTER_CARDS = (
    ('setup_strike', 'tremble', 'blood_wall'),
    ('breakthrough', 'inflame', 'anger'),
    ('iron_wave', 'dismantle', 'cinder'),
    ('stomp', 'shrug_it_off', 'armaments'),
    ('thunderclap', 'setup_strike', 'rage'),
    ('battle_trance', 'true_grit', 'uppercut'),
    ('bloodletting', 'whirlwind', 'tremble'),
)
MONSTER_POTIONS = (None, None, 'fire_potion', None, 'strength_potion', None, 'energy_potion')
ELITE_CARDS = (('bludgeon', 'pyre', 'evil_eye'), ('pillage', 'rampage', 'flame_barrier'))
ELITE_RELICS = ('vajra', 'ornamental_fan')
BOSS_CARDS = ('primal_force', 'demon_form', 'thrash')


@dataclass(frozen=True)
class TutorialRewardSpec:
    encounter: EncounterType
    index: int
    cards: tuple[str, ...]
    potion_id: str | None = None
    relic_id: str | None = None


def is_tutorial_profile(state: RunState) -> bool:
    """Missing profile data means the simulator's default all-unlocked profile."""
    if (not is_source_rng(state.rng) or state.character_id.upper() != 'IRONCLAD'
            or state.flags.get('number_of_runs') != 0):
        return False
    epochs = state.flags.get('revealed_epochs')
    return isinstance(epochs, Sequence) and not isinstance(epochs, str | bytes) and len(epochs) == 0


def _history(state: RunState) -> dict[str, dict[str, Any]]:
    raw = state.flags.get(HISTORY)
    if isinstance(raw, Mapping):
        return {str(key): dict(value) for key, value in raw.items() if isinstance(value, Mapping)}
    # Replay-backed older saves contain both original map points and actual
    # nested combat room types, including unknown-room outcomes.
    history: dict[str, dict[str, Any]] = {}
    for entry in state.replay_log:
        for event in entry.events:
            if event.kind not in {'room_entered', 'combat_started'} or not event.target_id:
                continue
            key = f"{event.metadata.get('act', state.act)}:{event.target_id}"
            value = history.setdefault(key, {'map_point_type': '', 'room_types': ()})
            kind = str(event.metadata.get('room_kind', ''))
            if event.kind == 'room_entered':
                value['map_point_type'] = kind
            else:
                value['room_types'] = tuple(dict.fromkeys((*value['room_types'], kind)))
    if state.map is not None:
        outcomes = state.flags.get('unknown_room_outcomes', {})
        for identifier in state.map.completed_node_ids:
            node = state.map.node_by_id.get(identifier)
            if node is None:
                continue
            key = f'{node.act}:{identifier}'
            if key not in history:
                kind = str(outcomes.get(identifier, node.kind.value)) if isinstance(
                    outcomes, Mapping) else node.kind.value
                history[key] = {'map_point_type': node.kind.value, 'room_types': (kind,)}
    return history


def record_combat_entry(state: RunState, node: MapNodeState) -> RunState:
    """Count a map point once even when its event enters multiple combats."""
    if not is_tutorial_profile(state):
        return state
    point = (state.map.node_by_id.get(state.map.current_node_id or node.node_id)
             if state.map is not None else None)
    point = point or node
    key = f'{point.act}:{point.node_id}'
    history = _history(state)
    previous = history.get(key, {})
    history[key] = {'map_point_type': point.kind.value, 'room_types': tuple(dict.fromkeys((
        *previous.get('room_types', ()), node.kind.value,
    )))}
    return state.model_copy(update={'flags': {**state.flags, HISTORY: history}})


def tutorial_reward_spec(state: RunState, encounter: EncounterType) -> TutorialRewardSpec | None:
    if not is_tutorial_profile(state):
        return None
    history = _history(state)
    if encounter is EncounterType.NORMAL:
        index = sum(RoomKind.MONSTER.value in value.get('room_types', ())
                    for value in history.values()) - 1
        if 0 <= index < len(MONSTER_CARDS):
            return TutorialRewardSpec(encounter, index, MONSTER_CARDS[index],
                                      MONSTER_POTIONS[index])
    elif encounter is EncounterType.ELITE:
        index = sum(value.get('map_point_type') == RoomKind.ELITE.value
                    for value in history.values()) - 1
        if 0 <= index < len(ELITE_CARDS):
            return TutorialRewardSpec(encounter, index, ELITE_CARDS[index],
                'block_potion' if index == 0 else None, ELITE_RELICS[index])
    elif encounter is EncounterType.BOSS:
        count = sum(value.get('map_point_type') == RoomKind.BOSS.value
                    for value in history.values())
        if count == 1:
            return TutorialRewardSpec(encounter, 0, BOSS_CARDS)
    return None


def draw_tutorial_bundle(
    state: RunState, rng: Random, spec: TutorialRewardSpec, pity: RewardPityState,
) -> CombatRewardBundle:
    """Populate only the fixed base set; shared orchestration owns all extras."""
    from . import transitions as t

    # Monster tutorial rewards explicitly construct GoldReward(10, 20), which
    # bypasses EncounterModel's Poverty endpoints and escape proportion.
    gold = (rng.randint(10, 20) if spec.encounter is EncounterType.NORMAL else _roll_gold(
        rng, CombatRewardContext(character_id=state.character_id, encounter=spec.encounter,
            act=state.act, ascension_level=state.ascension), DEFAULT_COMBAT_REWARD_RULES))
    rarities = tuple(str(t._reward_card_spec(state, identifier).get('rarity', 'common')).lower()
                     for identifier in spec.cards)
    return CombatRewardBundle(
        gold=gold, base_gold=gold, card_ids=spec.cards, card_rarities=rarities,
        card_upgraded=(False,) * len(spec.cards), potion_id=spec.potion_id,
        relic_ids=(spec.relic_id,) if spec.relic_id else (),
        relic_rarities=(('common' if spec.index == 0 else 'uncommon'),) if spec.relic_id else (),
        pity_state=pity,
    )


def tutorial_reward_metadata(spec: TutorialRewardSpec) -> dict[str, Any]:
    return {
        'tutorial_rewards': {'encounter': spec.encounter.value, 'index': spec.index},
        'card_reroll_contexts': {'primary': card_reroll_context(
            mode='combat', encounter=spec.encounter.value, pools=('ironclad',), count=3,
            initial_is_card_reward=True, is_card_reward=False, updates_on_relic_obtain=False,
        )},
    }

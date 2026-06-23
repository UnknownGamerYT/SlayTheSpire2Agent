"""Pure source-card special effect classifiers.

This module recognizes bounded card text patterns that are not ordinary source
fields yet. It emits deterministic marker steps for effects that can be wired
later, and explicit blockers for effects that require choices, timing hooks, or
combat state the pure source pass does not have.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

CardSpecialStatus = Literal["none", "executable", "explicit_blocker", "partial"]

CARD_SPECIAL_EFFECT_KEYS = frozenset(
    {
        "add_card_to_discard",
        "add_card_to_draw",
        "add_card_to_exhaust",
        "add_card_to_hand",
        "add_keyword_to_matching_cards",
        "add_keyword_to_random_card",
        "add_random_card_to_hand",
        "apply_status",
        "ally_channel_orb",
        "block_formula",
        "block_from_ally",
        "channel_orb",
        "combat_trigger",
        "choose_card",
        "damage_formula",
        "draw_formula",
        "dynamic_channel_orb",
        "evoke_orb",
        "explicit_blocker",
        "if_kill_resource",
        "orb_slot_delta",
        "osty_action",
        "player_resource",
        "self_cost_delta",
        "sovereign_blade",
        "status_formula",
        "timed_choice",
        "trigger_orb_passive",
    }
)


@dataclass(frozen=True, slots=True)
class CardSpecialPlan:
    """Special marker steps/events derived from one source card mapping."""

    card_id: str
    status: CardSpecialStatus
    steps: tuple[Mapping[str, Any], ...] = ()
    blockers: tuple[Mapping[str, Any], ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "card_id", _normalized_id(self.card_id) or "unknown_card")
        object.__setattr__(
            self,
            "steps",
            tuple(_clone_jsonish_mapping(step) for step in self.steps),
        )
        object.__setattr__(
            self,
            "blockers",
            tuple(_clone_jsonish_mapping(blocker) for blocker in self.blockers),
        )
        object.__setattr__(
            self,
            "events",
            tuple(_clone_jsonish_mapping(event) for event in self.events),
        )
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))


def card_special_plan(card_spec: Mapping[str, Any]) -> CardSpecialPlan:
    """Return bounded special marker steps and blockers for a source card."""

    card_id = _card_id(card_spec)
    description = _description_from(card_spec)
    description_text = _plain_text(description)
    sentences = _sentences(description)
    steps: list[Mapping[str, Any]] = []
    blockers: list[Mapping[str, Any]] = []
    events: list[Mapping[str, Any]] = []

    for sentence in sentences:
        _collect_if_kill_resource(
            sentence,
            card_id=card_id,
            steps=steps,
            events=events,
        )
        _collect_ally_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            events=events,
        )
        _collect_orb_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            blockers=blockers,
            events=events,
        )
        _collect_stance_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            blockers=blockers,
            events=events,
        )
        _collect_mantra_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            blockers=blockers,
            events=events,
        )
        _collect_forge_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            blockers=blockers,
            events=events,
        )
        _collect_summon_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            blockers=blockers,
            events=events,
        )
        _collect_osty_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            blockers=blockers,
            events=events,
        )
        _collect_soul_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            blockers=blockers,
            events=events,
        )
        _collect_sovereign_blade_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            blockers=blockers,
            events=events,
        )
        _collect_combat_trigger_markers(
            sentence,
            card_id=card_id,
            steps=steps,
            events=events,
        )
        _collect_choice_markers(
            sentence,
            card_id=card_id,
            context=description_text,
            steps=steps,
            blockers=blockers,
            events=events,
        )

    return CardSpecialPlan(
        card_id=card_id,
        status=_plan_status(steps, blockers),
        steps=tuple(steps),
        blockers=tuple(blockers),
        events=tuple(events),
        reasons=_blocker_reasons(blockers),
    )


def normalize_card_special_steps(card_spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return only executable special marker steps for a source card."""

    return card_special_plan(card_spec).steps


def card_special_events(card_spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return normalization/blocker events for a source card's special text."""

    return card_special_plan(card_spec).events


def card_special_blockers(card_spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return explicit blockers for special text that needs non-pure context."""

    return card_special_plan(card_spec).blockers


def _collect_ally_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    if "another player" not in sentence:
        return

    if ("channel" in sentence or "channels" in sentence) and "plasma" in sentence:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="ally_channel_orb",
            effect_key="ally_channel_orb",
            payload={"orb": "plasma", "amount": 1, "ally": "first"},
        )
    if "gain block equal to" in sentence and "block on another player" in sentence:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="block_from_ally",
            effect_key="block_from_ally",
            payload={"ally": "first", "field": "block"},
        )


def _collect_if_kill_resource(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    for resource, amount in re.findall(
        r"\bif\b[^.]*kills? an enemy[^.]*gain\s+\[(energy|star):(\d+)\]",
        sentence,
    ):
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="if_kill_resource",
            effect_key="if_kill_resource",
            payload={
                "resource": resource,
                "amount": int(amount),
                "trigger": "card_kills_enemy",
            },
        )


def _collect_orb_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    for match in re.finditer(r"\b(gain|lose)\s+(\d+)\s+orb slots?\b", sentence):
        amount = int(match.group(2))
        if match.group(1) == "lose":
            amount *= -1
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="orb_slot_delta",
            effect_key="orb_slot_delta",
            payload={"amount": amount},
        )

    if "channel" in sentence:
        timed_channel = _timed_channel_orb_payload(sentence)
        if timed_channel is not None:
            _append_step(
                steps,
                events,
                card_id=card_id,
                special=str(timed_channel["trigger"]),
                effect_key="combat_trigger",
                payload=timed_channel,
            )
        elif _has_dynamic_orb_channel_count(sentence):
            dynamic_channel = _dynamic_channel_orb_payload(sentence)
            if dynamic_channel is not None:
                _append_step(
                    steps,
                    events,
                    card_id=card_id,
                    special="dynamic_channel_orb",
                    effect_key="dynamic_channel_orb",
                    payload=dynamic_channel,
                )
            else:
                _append_blocker(
                    blockers,
                    events,
                    card_id=card_id,
                    kind="dynamic_orb_channel_count",
                    reason="requires combat state to determine how many orbs to channel",
                    metadata={"text": sentence},
                )
        elif "channel lightning equal to the lightning already channeled" in sentence:
            _append_step(
                steps,
                events,
                card_id=card_id,
                special="dynamic_channel_orb",
                effect_key="dynamic_channel_orb",
                payload={
                    "orb": "lightning",
                    "formula": "orb_channeled_this_combat",
                    "orb_filter": "lightning",
                },
            )
        else:
            for amount, orb in re.findall(
                r"\bchannel\s+(\d+)\s+(random orb|lightning|frost|dark|plasma|glass)\b",
                sentence,
            ):
                _append_step(
                    steps,
                    events,
                    card_id=card_id,
                    special="channel_orb",
                    effect_key="channel_orb",
                    payload={"orb": _normalized_orb(orb), "amount": int(amount)},
                )

    if "evoke" in sentence:
        timed_evoke = _timed_evoke_orb_payload(sentence)
        if timed_evoke is not None:
            _append_step(
                steps,
                events,
                card_id=card_id,
                special=str(timed_evoke["trigger"]),
                effect_key="combat_trigger",
                payload=timed_evoke,
            )
        elif "evoke all of your orbs" in sentence:
            _append_step(
                steps,
                events,
                card_id=card_id,
                special="evoke_orb",
                effect_key="evoke_orb",
                payload=_static_evoke_orb_payload(sentence) or {
                    "selector": "all",
                    "amount": "all",
                },
            )
        else:
            evoke_match = re.search(
                r"\bevoke your (leftmost|rightmost) orb(?:\s+(twice|(\d+)\s+times?))?",
                sentence,
            )
            if evoke_match:
                _append_step(
                    steps,
                    events,
                    card_id=card_id,
                    special="evoke_orb",
                    effect_key="evoke_orb",
                    payload={
                        "selector": evoke_match.group(1),
                        "amount": _repeat_amount(
                            evoke_match.group(2),
                            evoke_match.group(3),
                        ),
                    },
                )

    if "orb" in sentence and "for each unique orb" in sentence:
        unique_payload = _unique_orb_formula_payload(sentence)
        if unique_payload is not None:
            effect_key = str(unique_payload["effect_key"])
            _append_step(
                steps,
                events,
                card_id=card_id,
                special=effect_key,
                effect_key=effect_key,
                payload=unique_payload["payload"],
            )
        else:
            _append_blocker(
                blockers,
                events,
                card_id=card_id,
                kind="orb_state_scaling",
                reason="requires current orb state to scale the card effect",
                metadata={"text": sentence},
            )
    direct_passive_trigger = _direct_orb_passive_trigger_payload(sentence)
    if direct_passive_trigger is not None:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="trigger_orb_passive",
            effect_key="trigger_orb_passive",
            payload=direct_passive_trigger,
        )

    if "trigger the passive ability" in sentence and "orb" in sentence:
        passive_trigger = _orb_passive_trigger_payload(sentence)
        if passive_trigger is not None:
            if _is_timed_or_triggered(sentence):
                _append_step(
                    steps,
                    events,
                    card_id=card_id,
                    special="turn_start",
                    effect_key="combat_trigger",
                    payload={
                        "trigger": "turn_start",
                        "duration": "combat",
                        "effects": ({"trigger_orb_passive": passive_trigger},),
                        "text": sentence,
                    },
                )
            else:
                _append_step(
                    steps,
                    events,
                    card_id=card_id,
                    special="trigger_orb_passive",
                    effect_key="trigger_orb_passive",
                    payload=passive_trigger,
                )
        else:
            _append_blocker(
                blockers,
                events,
                card_id=card_id,
                kind="orb_passive_trigger",
                reason="requires orb passive trigger execution",
                metadata={"text": sentence},
            )


def _collect_forge_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    if "forge" not in sentence:
        return
    if "whenever you forge" in sentence:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="forge_ally_trigger",
            effect_key="combat_trigger",
            payload={
                "trigger": "player_resource_changed",
                "duration": "combat",
                "condition": {"resource": "forge"},
                "effects": ({"forge_allies": {"source": "hammer_time"}},),
                "text": sentence,
            },
        )
        return
    if _is_timed_or_triggered(sentence):
        forge_match = re.search(r"\bforge\s+(\d+)\b", sentence)
        if forge_match and sentence.startswith("at the start of your turn"):
            _append_step(
                steps,
                events,
                card_id=card_id,
                special="turn_start",
                effect_key="combat_trigger",
                payload={
                    "trigger": "turn_start",
                    "duration": "combat",
                    "effects": (
                        {
                            "player_resource": {
                                "resource": "forge",
                                "amount": int(forge_match.group(1)),
                            }
                        },
                    ),
                    "text": sentence,
                },
            )
        else:
            _append_blocker(
                blockers,
                events,
                card_id=card_id,
                kind="timed_or_triggered_forge",
                reason="requires a combat timing hook before granting forge",
                metadata={"text": sentence},
            )
        return
    if "for every other time" in sentence:
        base_match = re.search(r"\b(?:forge|forges)\s+an additional\s+(\d+)\b", sentence)
        amount = int(base_match.group(1)) if base_match else 1
        _append_resource_step(
            steps,
            events,
            card_id=card_id,
            resource="forge",
            amount=0,
            special="dynamic_forge",
            amount_payload={
                "formula": "target_hits_this_turn",
                "multiplier": amount,
                "bonus": -amount,
            },
        )
        return
    if re.search(r"\bforge\s+x\b", sentence):
        _append_resource_step(
            steps,
            events,
            card_id=card_id,
            resource="forge",
            amount=0,
            special="dynamic_forge",
            amount_payload={"formula": "energy_spent"},
        )
        return
    for amount in re.findall(r"\bforge\s+(\d+)\b", sentence):
        _append_resource_step(
            steps,
            events,
            card_id=card_id,
            resource="forge",
            amount=int(amount),
            special="forge",
        )


def _collect_stance_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    if not _has_any_word(sentence, ("stance", "wrath", "calm", "divinity")):
        return

    if sentence.startswith(("if ", "while ")) and _has_any_word(
        sentence, ("stance", "wrath", "calm", "divinity")
    ):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="stance_state_condition",
            reason="requires checking the current stance before resolving the card",
            metadata={"text": sentence},
        )
        return

    if sentence.startswith(("at the start", "at the end", "when ", "whenever ")):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="timed_or_triggered_stance_effect",
            reason="requires a combat timing hook before changing or checking stance",
            metadata={"text": sentence},
        )
        return

    if "random stance" in sentence:
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="random_stance_selection",
            reason="requires deterministic random stance selection before entering stance",
            metadata={"text": sentence},
        )
        return

    stance_pattern = r"\benter\s+(?:your\s+)?(wrath|calm|divinity)(?:\s+stance)?\b"
    for stance in re.findall(stance_pattern, sentence):
        _append_step(
            steps,
            events,
            card_id=card_id,
            special=f"enter_{stance}",
            effect_key="apply_status",
            payload={"target": "self", f"stance_{stance}": 1},
        )

    if re.search(r"\b(?:exit|leave)\s+(?:your\s+)?stance\b", sentence):
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="exit_stance",
            effect_key="apply_status",
            payload={"target": "self", "stance_none": 1},
        )


def _collect_mantra_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    if "mantra" not in sentence:
        return
    if _is_timed_or_triggered(sentence):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="timed_or_triggered_mantra",
            reason="requires a combat timing hook before granting mantra",
            metadata={"text": sentence},
        )
        return
    if re.search(r"\bgain\s+x\s+mantra\b", sentence) or "for each" in sentence:
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="dynamic_mantra_amount",
            reason="requires combat state or X-cost energy spent to determine mantra",
            metadata={"text": sentence},
        )
        return
    for amount in re.findall(r"\bgain\s+(\d+)\s+mantra\b", sentence):
        _append_resource_step(
            steps,
            events,
            card_id=card_id,
            resource="mantra",
            amount=int(amount),
            special="mantra",
        )


def _collect_summon_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    if "summon" not in sentence:
        return
    if "whenever osty hits this enemy" in sentence:
        return
    if "whenever you play a soul" in sentence:
        return
    if _is_timed_or_triggered(sentence):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="timed_or_triggered_summon",
            reason="requires a combat timing hook before granting summon",
            metadata={"text": sentence},
        )
        return
    dynamic_match = re.search(r"\bsummon\s+(?:(\d+)\s+)?x\b", sentence)
    if dynamic_match:
        _append_resource_step(
            steps,
            events,
            card_id=card_id,
            resource="summon",
            amount=0,
            special="dynamic_summon",
            amount_payload={
                "amount": 0,
                "per_energy": int(dynamic_match.group(1) or 1),
            },
        )
        return
    for amount in re.findall(r"\bsummon\s+(\d+)\b", sentence):
        _append_resource_step(
            steps,
            events,
            card_id=card_id,
            resource="summon",
            amount=int(amount),
            special="summon",
        )


def _collect_osty_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    if "osty" not in sentence:
        return
    if sentence.startswith("if osty is alive") or sentence.startswith("if osty's alive"):
        payloads = _conditional_osty_payloads(sentence)
        if payloads:
            _append_step(
                steps,
                events,
                card_id=card_id,
                special="osty_action",
                effect_key="osty_action",
                payload=payloads,
            )
        return

    if "whenever osty loses hp" in sentence:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="osty_loss_damage",
            effect_key="osty_action",
            payload={"action": "enable_loss_damage"},
        )
        return

    for match in re.finditer(
        r"\bosty(?:'s)?\s+deals\s+(\d+)\s+damage(?:\s+to\s+(all enemies|a random enemy))?",
        sentence,
    ):
        target = _osty_target_from_sentence(sentence, match.group(2) or "")
        payload: dict[str, Any] = {
            "action": "damage",
            "amount": int(match.group(1)),
            "target": target,
        }
        payload.update(_osty_damage_modifiers(sentence))
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="osty_action",
            effect_key="osty_action",
            payload=payload,
        )

    heal_match = re.search(r"\bosty(?:'s)?\s+heals\s+(\d+)\s+hp\b", sentence)
    if heal_match:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="osty_action",
            effect_key="osty_action",
            payload={"action": "heal", "amount": int(heal_match.group(1))},
        )

    if re.search(r"\bosty(?:'s)?\s+dies\b", sentence):
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="osty_action",
            effect_key="osty_action",
            payload={"action": "die"},
        )

    if "osty's attacks deal" in sentence:
        modifier_match = re.search(r"\battacks\s+deal\s+(\d+)\s+additional\s+damage\b", sentence)
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="osty_damage_modifier",
            effect_key="osty_action",
            payload={
                "action": "modify_damage",
                "amount": int(modifier_match.group(1)) if modifier_match else 0,
            },
        )


def _collect_soul_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    if "soul" not in sentence:
        return
    if "whenever you play a soul" in sentence:
        trigger_effects: list[Mapping[str, Any]] = []
        summon_match = re.search(r"\bsummon\s+(\d+)\b", sentence)
        if summon_match:
            trigger_effects.append(
                {
                    "player_resource": {
                        "resource": "summon",
                        "amount": int(summon_match.group(1)),
                    }
                }
            )
        hp_loss_match = re.search(r"\brandom enemy loses\s+(\d+)\s+hp\b", sentence)
        if hp_loss_match:
            trigger_effects.append(
                {
                    "enemy_hp_loss": {
                        "target": "random_enemy",
                        "amount": int(hp_loss_match.group(1)),
                    }
                }
            )
        if trigger_effects:
            _append_step(
                steps,
                events,
                card_id=card_id,
                special="soul_play_trigger",
                effect_key="combat_trigger",
                payload={
                    "trigger": "card_played",
                    "duration": "combat",
                    "condition": {"card_id": "soul"},
                    "effects": tuple(trigger_effects),
                    "text": sentence,
                },
            )
        else:
            _append_blocker(
                blockers,
                events,
                card_id=card_id,
                kind="soul_play_trigger",
                reason="requires a card-play trigger for Soul cards",
                metadata={"text": sentence},
            )
    if "for each soul" in sentence:
        amount_match = re.search(r"\bdeals\s+(\d+)\s+additional\s+damage\b", sentence)
        if amount_match:
            _append_step(
                steps,
                events,
                card_id=card_id,
                special="soul_count_scaling",
                effect_key="damage_formula",
                payload={
                    "formula": "soul_count",
                    "zone": "exhaust_pile",
                    "multiplier": int(amount_match.group(1)),
                },
            )
        else:
            _append_blocker(
                blockers,
                events,
                card_id=card_id,
                kind="soul_count_scaling",
                reason="requires pile contents to scale the card effect",
                metadata={"text": sentence},
            )
    if not sentence.startswith("add ") or " into your " not in sentence:
        return

    count_match = re.search(r"\badd\s+(\d+|a|an|x)\s+souls?\+?\b", sentence)
    if count_match is None:
        return
    raw_count = count_match.group(1)
    if raw_count == "x":
        upgraded = "soul+" in sentence or "upgraded soul" in sentence
        for destination in _destinations_from_sentence(sentence):
            _append_step(
                steps,
                events,
                card_id=card_id,
                special="dynamic_add_soul",
                effect_key=f"add_card_to_{destination}",
                payload={
                    **_soul_card_payload(upgraded=upgraded),
                    "count_formula": {"formula": "energy_spent"},
                },
            )
        return

    amount = _word_amount(raw_count)
    upgraded = "soul+" in sentence or "upgraded soul" in sentence
    for destination in _destinations_from_sentence(sentence):
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="add_soul",
            effect_key=f"add_card_to_{destination}",
            payload=tuple(_soul_card_payload(upgraded=upgraded) for _ in range(amount)),
            amount=amount,
        )


def _collect_sovereign_blade_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    if "sovereign blade" not in sentence:
        return
    if "deals double damage" in sentence:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="sovereign_blade_modifier",
            effect_key="sovereign_blade",
            payload={
                "action": "temporary_modifier",
                "modifier": "double_damage",
                "duration": "turn",
                "target": "enemy",
            },
        )
    if "now deals damage to all enemies" in sentence:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="sovereign_blade_modifier",
            effect_key="sovereign_blade",
            payload={"action": "set_target", "target": "all_enemies"},
        )
    if "now hits an additional time" in sentence:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="sovereign_blade_modifier",
            effect_key="sovereign_blade",
            payload={"action": "add_hit", "amount": 1},
        )
    block_match = re.search(r"\bsovereign blade now gains\s+(\d+)\s+block\b", sentence)
    if block_match:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="sovereign_blade_modifier",
            effect_key="sovereign_blade",
            payload={"action": "gain_block", "amount": int(block_match.group(1))},
        )
    replay_match = re.search(r"\bsovereign blade gains replay\s+(\d+)\b", sentence)
    if replay_match:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="sovereign_blade_replay",
            effect_key="add_keyword_to_matching_cards",
            payload={
                "keyword": "replay",
                "amount": int(replay_match.group(1)),
                "filter": {"card_id_contains": "sovereign_blade", "exclude_keyword": "replay"},
                "zones": ("hand", "draw_pile", "discard_pile", "exhaust_pile"),
            },
        )
    if "put sovereign blade into your hand from anywhere" in sentence:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="sovereign_blade_modifier",
            effect_key="sovereign_blade",
            payload={"action": "move_to_hand", "source": "anywhere"},
        )
    if "whenever you play sovereign blade" in sentence:
        block_match = re.search(r"\bgain\s+(\d+)\s+block\b", sentence)
        _append_step(
            steps,
            events,
            card_id=card_id,
            special="sovereign_blade_play_trigger",
            effect_key="combat_trigger",
            payload={
                "trigger": "card_played",
                "duration": "combat",
                "condition": {"card_id": "sovereign_blade"},
                "effects": ({"block": int(block_match.group(1)) if block_match else 0},),
                "text": sentence,
            },
        )
    if "sovereign blade" in sentence and not any(
        phrase in sentence
        for phrase in (
            "deals double damage",
            "now deals damage to all enemies",
            "now hits an additional time",
            "now gains",
            "gains replay",
            "put sovereign blade into your hand from anywhere",
            "whenever you play sovereign blade",
        )
    ):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="sovereign_blade_unknown_special",
            reason="requires a Sovereign Blade special handler for this text",
            metadata={"text": sentence},
        )


def _collect_choice_markers(
    sentence: str,
    *,
    card_id: str,
    context: str,
    steps: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    timed_payload = _timed_choice_payload(sentence, context=context)
    if timed_payload is not None:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special=str(timed_payload["trigger"]),
            effect_key="timed_choice",
            payload=timed_payload,
        )
        return

    payload = _supported_choice_payload(sentence, context=context)
    if payload is not None:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special=str(payload["action"]),
            effect_key="choose_card",
            payload=payload,
            amount=int(payload.get("count", 1)),
        )
        return

    if "choose" in sentence:
        discovery = "random" in sentence and _choice_source_zone(sentence) is None
        kind = "card_discovery_required" if discovery else "choose_card_required"
        reason = (
            "requires presenting generated card choices before the card can resolve"
            if discovery
            else "requires choosing card targets from a combat zone before the card can resolve"
        )
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind=kind,
            reason=reason,
            metadata={
                "text": sentence,
                "zone": _choice_source_zone(sentence),
                "count": _choice_count(sentence),
                "choices": _choice_pool_count(sentence),
                "action": _choice_action(sentence),
            },
        )
        return

    if _requires_implicit_card_choice(sentence):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="choose_card_required",
            reason="requires choosing card targets from a combat zone before the card can resolve",
            metadata={
                "text": sentence,
                "zone": _choice_source_zone(sentence),
                "count": _choice_count(sentence),
                "action": _choice_action(sentence),
            },
        )


def _collect_combat_trigger_markers(
    sentence: str,
    *,
    card_id: str,
    steps: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    for special, effect_key, payload in _direct_special_effect_payloads(sentence):
        _append_step(
            steps,
            events,
            card_id=card_id,
            special=special,
            effect_key=effect_key,
            payload=payload,
        )

    status_payload = _combat_modifier_status_payload(sentence)
    if status_payload is not None:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special=str(status_payload["special"]),
            effect_key="apply_status",
            payload=status_payload["payload"],
        )
        return

    trigger_payload = _combat_trigger_payload(sentence, card_id=card_id)
    if trigger_payload is not None:
        _append_step(
            steps,
            events,
            card_id=card_id,
            special=str(trigger_payload["trigger"]),
            effect_key="combat_trigger",
            payload=trigger_payload,
        )


def _direct_special_effect_payloads(sentence: str) -> tuple[tuple[str, str, Any], ...]:
    payloads: list[tuple[str, str, Any]] = []

    sandpit_match = re.search(r"\bincrease\s+sandpit\s+by\s+(\d+)\b", sentence)
    if sandpit_match:
        payloads.append(
            (
                "sandpit_gain",
                "player_resource",
                {
                    "resource": "sandpit",
                    "amount": int(sandpit_match.group(1)),
                    "source": "card_special",
                },
            )
        )

    cost_match = re.search(r"\bincrease the cost of this card by\s+(\d+)\b", sentence)
    if cost_match:
        payloads.append(("self_cost_delta", "self_cost_delta", int(cost_match.group(1))))

    replay_match = re.search(
        r"\brandom card without replay in your draw pile gains replay\s+(\d+)\b",
        sentence,
    )
    if replay_match:
        payloads.append(
            (
                "random_draw_card_replay",
                "add_keyword_to_random_card",
                {
                    "zone": "draw_pile",
                    "keyword": "replay",
                    "amount": int(replay_match.group(1)),
                    "exclude_keyword": "replay",
                },
            )
        )

    if "another player adds" in sentence and "random colorless card" in sentence:
        upgraded = "upgraded" in sentence
        payloads.append(
            (
                "ally_random_colorless_card",
                "add_random_card_to_hand",
                {
                    "count": 1,
                    "pool": "colorless",
                    "upgraded": upgraded,
                    "source": "card_special",
                },
            )
        )

    return tuple(payloads)


def _append_resource_step(
    steps: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
    *,
    card_id: str,
    resource: str,
    amount: int,
    special: str,
    amount_payload: Mapping[str, Any] | None = None,
) -> None:
    _append_step(
        steps,
        events,
        card_id=card_id,
        special=special,
        effect_key="player_resource",
        payload={
            "resource": resource,
            "amount": _clone_jsonish(amount_payload) if amount_payload is not None else amount,
            "source": "card_special",
        },
    )


def _append_step(
    steps: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
    *,
    card_id: str,
    special: str,
    effect_key: str,
    payload: Any,
    amount: int | None = None,
) -> None:
    step = {effect_key: _clone_jsonish(payload)}
    if step in steps:
        return
    steps.append(step)
    events.append(
        {
            "kind": "card_special_normalized",
            "source_id": card_id,
            "target_id": None,
            "amount": amount if amount is not None else _payload_amount(payload),
            "metadata": {
                "classification": "executable",
                "special": special,
                "effect": _clone_jsonish_mapping(step),
            },
        }
    )


def _append_blocker(
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
    *,
    card_id: str,
    kind: str,
    reason: str,
    metadata: Mapping[str, Any],
) -> None:
    payload = {
        "kind": kind,
        "reason": reason,
        "metadata": _clone_jsonish_mapping(metadata),
    }
    blocker = {"explicit_blocker": payload}
    if blocker in blockers:
        return
    blockers.append(blocker)
    events.append(
        {
            "kind": "card_special_blocker",
            "source_id": card_id,
            "target_id": None,
            "amount": _payload_amount(metadata),
            "message": reason,
            "metadata": {
                "classification": "explicit_blocker",
                "special": kind,
                "reason": reason,
                "blocker": _clone_jsonish_mapping(payload),
            },
        }
    )


def _description_from(card_spec: Mapping[str, Any]) -> str:
    if bool(card_spec.get("upgraded")) and card_spec.get("upgrade_description"):
        return str(card_spec.get("upgrade_description") or "")
    return str(card_spec.get("description", card_spec.get("description_raw", "")) or "")


def _sentences(description: str) -> tuple[str, ...]:
    text = _plain_text(description)
    parts = re.split(r"(?:\n|(?<=[.!?])\s+)", text.replace("\n", ". "))
    return tuple(part.strip(" .") for part in parts if part.strip(" ."))


def _plain_text(description: str) -> str:
    text = re.sub(r"\[/?(?!energy:|star:)[^\]]+\]", "", description)
    return " ".join(text.lower().split())


def _is_timed_or_triggered(sentence: str) -> bool:
    return sentence.startswith(
        (
            "at the start",
            "at the end",
            "if ",
            "when ",
            "whenever ",
        )
    )


def _has_dynamic_orb_channel_count(sentence: str) -> bool:
    return any(
        phrase in sentence
        for phrase in (
            "channel x",
            "for each enemy",
            "equal to",
            "already channeled",
        )
    )


def _timed_channel_orb_payload(sentence: str) -> Mapping[str, Any] | None:
    if "channel" not in sentence:
        return None
    channel = _static_channel_orb_payload(sentence)
    if channel is None:
        return None
    if sentence.startswith("whenever you play a power"):
        return {
            "trigger": "card_played",
            "duration": "combat",
            "condition": {"card_type": "power"},
            "effects": ({"channel_orb": channel},),
            "text": sentence,
        }
    if sentence.startswith("at the start of your turn"):
        return {
            "trigger": "turn_start",
            "duration": "combat",
            "effects": ({"channel_orb": channel},),
            "text": sentence,
        }
    next_turns = re.search(r"\bat the start of the next\s+(\d+)\s+turns?", sentence)
    if next_turns:
        return {
            "trigger": "turn_start",
            "duration": "once",
            "remaining_uses": int(next_turns.group(1)),
            "effects": ({"channel_orb": channel},),
            "text": sentence,
        }
    return None


def _static_channel_orb_payload(sentence: str) -> Mapping[str, Any] | None:
    match = re.search(
        r"\bchannel\s+(?:(\d+)\s+)?(random orb|lightning|frost|dark|plasma|glass)\b",
        sentence,
    )
    if not match:
        return None
    return {
        "orb": _normalized_orb(match.group(2)),
        "amount": int(match.group(1) or 1),
    }


def _dynamic_channel_orb_payload(sentence: str) -> Mapping[str, Any] | None:
    orb_match = re.search(
        r"\bchannel\s+(?:x(?:\+(\d+))?|\d+)?\s*"
        r"(random orb|lightning|frost|dark|plasma|glass)\b",
        sentence,
    )
    orb = _normalized_orb(orb_match.group(2)) if orb_match else "lightning"
    if "for each enemy" in sentence:
        return {"orb": orb, "formula": "alive_enemy_count"}
    if "channel x" in sentence:
        bonus = int(orb_match.group(1) or 0) if orb_match else 0
        payload: dict[str, Any] = {"orb": orb, "formula": "energy_spent"}
        if bonus:
            payload["bonus"] = bonus
        return payload
    if "already channeled" in sentence:
        return {
            "orb": orb,
            "formula": "orb_channeled_this_combat",
            "orb_filter": orb,
        }
    return None


def _timed_evoke_orb_payload(sentence: str) -> Mapping[str, Any] | None:
    if "evoke" not in sentence:
        return None
    evoke = _static_evoke_orb_payload(sentence)
    if evoke is None:
        return None
    if sentence.startswith("at the end of your turn"):
        return {
            "trigger": "turn_end",
            "duration": "combat",
            "effects": ({"evoke_orb": evoke},),
            "text": sentence,
        }
    return None


def _static_evoke_orb_payload(sentence: str) -> Mapping[str, Any] | None:
    if "evoke all of your orbs" in sentence:
        amount: int | str = "all"
        repeat = re.search(r"\bevoke all of your orbs\s+(twice|(\d+)\s+times?)", sentence)
        if repeat:
            amount = _repeat_amount(repeat.group(1), repeat.group(2))
        return {"selector": "all", "amount": amount}
    match = re.search(
        r"\bevoke your (leftmost|rightmost) orb(?:\s+(twice|(\d+)\s+times?))?",
        sentence,
    )
    if not match:
        return None
    return {
        "selector": match.group(1),
        "amount": _repeat_amount(match.group(2), match.group(3)),
    }


def _direct_orb_passive_trigger_payload(sentence: str) -> Mapping[str, Any] | None:
    match = re.search(
        r"\btrigger all\s+(lightning|frost|dark|plasma|glass)\b",
        sentence,
    )
    if not match:
        return None
    amount = 2 if "twice" in sentence else 1
    repeat_match = re.search(r"\b(\d+)\s+times?\b", sentence)
    if repeat_match:
        amount = int(repeat_match.group(1))
    orb = _normalized_orb(match.group(1))
    return {"selector": "all", "amount": amount, "orb_filter": orb}


def _unique_orb_formula_payload(sentence: str) -> Mapping[str, Any] | None:
    amount_match = re.search(r"\b(?:draw|gain)\s+(\d+)\b", sentence)
    multiplier = int(amount_match.group(1)) if amount_match else 1
    if "draw" in sentence:
        return {
            "effect_key": "draw_formula",
            "payload": {"formula": "unique_orb_count", "multiplier": multiplier},
        }
    if "block" in sentence:
        return {
            "effect_key": "block_formula",
            "payload": {"formula": "unique_orb_count", "multiplier": multiplier},
        }
    if "focus" in sentence:
        return {
            "effect_key": "status_formula",
            "payload": {
                "target": "self",
                "status": "temporary_focus" if "this turn" in sentence else "focus",
                "formula": "unique_orb_count",
                "multiplier": multiplier,
            },
        }
    return None


def _orb_passive_trigger_payload(sentence: str) -> Mapping[str, Any] | None:
    amount = 2 if "twice" in sentence else (_first_int(sentence) or 1)
    if "all" in sentence:
        payload: dict[str, Any] = {"selector": "all", "amount": amount}
        orb_match = re.search(r"\ball\s+(lightning|frost|dark|plasma|glass)\s+orbs?\b", sentence)
        if orb_match:
            payload["orb_filter"] = _normalized_orb(orb_match.group(1))
        return payload
    selector = "rightmost" if "rightmost" in sentence else "leftmost"
    return {"selector": selector, "amount": amount}


def _repeat_amount(raw_repeat: str | None, raw_number: str | None) -> int:
    if raw_repeat == "twice":
        return 2
    if raw_number is not None:
        return int(raw_number)
    return 1


def _normalized_orb(orb: str) -> str:
    key = _normalized_id(orb)
    if key == "random_orb":
        return "random"
    return key


def _conditional_osty_payloads(sentence: str) -> tuple[Mapping[str, Any], ...]:
    payloads: list[Mapping[str, Any]] = []
    damage_match = re.search(
        r"\b(?:osty|he)\s+deals\s+(\d+)\s+damage(?:\s+to\s+(all enemies|a random enemy))?",
        sentence,
    )
    if damage_match:
        payloads.append(
            {
                "action": "damage",
                "amount": int(damage_match.group(1)),
                "target": _osty_target_from_sentence(sentence, damage_match.group(2) or ""),
            }
        )
    block_match = re.search(r"\bgain\s+(\d+)\s+block\b", sentence)
    if block_match:
        payloads.append({"action": "block", "amount": int(block_match.group(1))})
    if "gain block equal to double his max hp" in sentence:
        payloads.append({"action": "block", "amount": "double_max_hp"})
    if re.search(r"\b(?:osty|he)\s+dies\b", sentence):
        payloads.append({"action": "die"})
    return tuple(payloads)


def _osty_damage_modifiers(sentence: str) -> dict[str, Any]:
    modifiers: dict[str, Any] = {}
    if "additional damage equal to osty's max hp" in sentence:
        modifiers["bonus"] = "max_hp"
    elif "additional damage equal to osty's current hp" in sentence:
        modifiers["bonus"] = "current_hp"
    elif "for all your other osty attacks" in sentence:
        modifiers["bonus"] = "other_osty_attacks"
        amount_match = re.search(r"\bdeals\s+(\d+)\s+additional\s+damage\s+for\b", sentence)
        modifiers["bonus_per"] = int(amount_match.group(1)) if amount_match else 1
    if "hits an additional time for each other time he has attacked this turn" in sentence:
        modifiers["extra_hits"] = "attacks_this_turn"
    return modifiers


def _osty_target_from_sentence(sentence: str, raw_target: str) -> str:
    if "to a random enemy" in sentence:
        return "random_enemy"
    if "all enemies" in sentence:
        return "all_enemies"
    return _osty_target(raw_target)


def _osty_target(raw_target: str) -> str:
    key = _normalized_id(raw_target)
    if key == "all_enemies":
        return "all_enemies"
    if key == "a_random_enemy":
        return "random_enemy"
    return "enemy"


def _destinations_from_sentence(sentence: str) -> tuple[str, ...]:
    destinations: list[str] = []
    for phrase, destination in (
        ("draw pile", "draw"),
        ("hand", "hand"),
        ("discard pile", "discard"),
        ("exhaust pile", "exhaust"),
    ):
        if phrase in sentence and destination not in destinations:
            destinations.append(destination)
    return tuple(destinations or ("hand",))


def _soul_card_payload(*, upgraded: bool) -> Mapping[str, Any]:
    draw = 3 if upgraded else 2
    return {
        "card": {
            "id": "SOUL",
            "name": "Soul+" if upgraded else "Soul",
            "type": "Skill",
            "target": "Self",
            "cost": 1,
            "draw": draw,
            "description": f"Draw {draw} cards.",
            "keywords_key": ("Exhaust",),
            "exhausts": True,
            "upgraded": upgraded,
        },
        "temporary": True,
    }


def _supported_choice_payload(sentence: str, *, context: str) -> Mapping[str, Any] | None:
    if _is_timed_or_triggered(sentence):
        return None

    action = _choice_action(sentence)
    zone = _choice_source_zone(sentence)
    count = _choice_count(sentence) or 1
    choices = _choice_pool_count(sentence)
    card_types = _choice_card_types(sentence)

    if action == "select" and _choice_is_next_turn_copy(sentence, context):
        next_turn_copy_payload: dict[str, Any] = {
            "action": "copy_to_hand_next_turn",
            "zone": zone or "hand",
            "count": count,
            "copy_count": _next_turn_copy_count(context),
            "destination": "hand",
            "text": sentence,
        }
        if card_types:
            next_turn_copy_payload["card_types"] = card_types
        return next_turn_copy_payload

    if (
        action == "move_to_hand"
        and "random" in sentence
        and choices is not None
    ):
        generated_payload: dict[str, Any] = {
            "action": action,
            "zone": "generated",
            "count": count,
            "choices": choices,
            "destination": "hand",
            "pool": _generated_choice_pool(sentence),
            "free_to_play_this_turn": "free to play this turn" in context,
            "text": sentence,
        }
        if card_types:
            generated_payload["card_types"] = card_types
        return generated_payload

    if action == "select" and _choice_is_copy_to_hand(sentence):
        copy_payload: dict[str, Any] = {
            "action": "copy_to_hand",
            "zone": zone or "hand",
            "count": count,
            "copy_count": 1,
            "destination": "hand",
            "text": sentence,
        }
        if card_types:
            copy_payload["card_types"] = card_types
        if "colorless" in sentence:
            copy_payload["color"] = "colorless"
        return copy_payload

    if count <= 0 or zone is None:
        return None
    if "random" in sentence and choices is not None:
        return None

    payload: dict[str, Any] = {
        "action": action,
        "zone": zone,
        "count": count,
        "text": sentence,
    }
    if choices is not None:
        payload["choices"] = choices
    if card_types:
        payload["card_types"] = card_types

    if action == "play" and zone == "hand":
        payload["play_times"] = _choice_play_times(sentence)
        return payload
    if action == "add_retain" and zone == "hand":
        return payload
    if action == "exhaust" and zone in {"hand", "draw_pile", "discard_pile"}:
        return payload
    if action == "move_to_hand" and zone in {"draw_pile", "discard_pile"}:
        payload["destination"] = "hand"
        return payload
    if action == "move_to_draw_top" and zone in {"hand", "discard_pile"}:
        payload["destination"] = "draw_pile_top"
        return payload
    if action == "transform" and zone in {"hand", "draw_pile", "discard_pile"}:
        target = _transform_target_card_id(sentence)
        if target is None:
            return None
        payload["target_card_id"] = target
        return payload
    return None


def _timed_choice_payload(sentence: str, *, context: str) -> Mapping[str, Any] | None:
    if (
        sentence.startswith("at the start")
        and "your turn" in sentence
        and "transform" in sentence
        and "card in your hand" in sentence
    ):
        return {
            "trigger": "turn_start",
            "repeat": True,
            "choose_card": {
                "action": "transform",
                "zone": "hand",
                "count": _choice_count(sentence) or 1,
                "random_transform": True,
                "text": sentence,
            },
            "text": sentence,
        }

    if (
        sentence.startswith("at the start")
        and "your turn" in sentence
        and "draw" in sentence
        and "discard" in sentence
        and "card" in sentence
    ):
        return {
            "trigger": "turn_start",
            "repeat": True,
            "pre_effects": ({"draw": _draw_amount_from_sentence(sentence)},),
            "choose_card": {
                "action": "discard",
                "zone": "hand",
                "count": _choice_count(sentence) or 1,
                "text": sentence,
            },
            "text": sentence,
        }

    if (
        sentence.startswith("at the start")
        and "your turn" in sentence
        and "draw" in sentence
        and "exhaust" in sentence
        and "card from your hand" in sentence
    ):
        return {
            "trigger": "turn_start",
            "repeat": True,
            "pre_effects": ({"draw": _draw_amount_from_sentence(sentence)},),
            "choose_card": {
                "action": "exhaust",
                "zone": "hand",
                "count": _choice_count(sentence) or 1,
                "text": sentence,
            },
            "text": sentence,
        }

    if (
        sentence.startswith("whenever")
        and "shuffle your draw pile" in sentence
        and "choose" in sentence
        and "put" in sentence
        and "hand" in sentence
    ):
        return {
            "trigger": "draw_pile_shuffled",
            "repeat": True,
            "choose_card": {
                "action": "move_to_hand",
                "zone": "draw_pile",
                "count": _choice_count(sentence) or 1,
                "destination": "hand",
                "text": sentence,
            },
            "text": sentence,
        }

    return None


def _combat_modifier_status_payload(sentence: str) -> Mapping[str, Any] | None:
    if "whenever you spend" in sentence and "star" in sentence and "block" in sentence:
        amount = _first_int_after(sentence, "gain") or _last_int(sentence) or 1
        return {
            "special": "child_of_the_stars",
            "payload": {"target": "self", "child_of_the_stars": amount},
        }
    if "the enemy takes double attack damage from other players this turn" in sentence:
        return {
            "special": "flanking",
            "payload": {"target": "enemy", "flanking": 1},
        }
    if "whenever attacks deal damage" in sentence and "doom" in sentence:
        return {
            "special": "reaper_form",
            "payload": {"target": "self", "reaper_form": 1},
        }
    if (
        "whenever you attack an enemy" in sentence
        and "loses" in sentence
        and "strength" in sentence
    ):
        amount = _first_int(sentence) or 1
        return {
            "special": "monarchs_gaze",
            "payload": {"target": "self", "monarchs_gaze": amount},
        }
    if "take double damage from enemies" in sentence:
        return {
            "special": "tank",
            "payload": {"target": "self", "tank": 1},
        }
    if "weak enemies take double damage from attacks" in sentence:
        return {
            "special": "tracking",
            "payload": {"target": "self", "tracking": 1},
        }
    if "block is not removed at the start of your turn" in sentence:
        return {
            "special": "retain_block",
            "payload": {"target": "self", "retain_block": 1},
        }
    if "double your block gain this turn" in sentence:
        return {
            "special": "double_block_gain_this_turn",
            "payload": {"target": "self", "double_block_gain_this_turn": 1},
        }
    block_bonus_match = re.search(
        r"\bgain an additional\s+(\d+)\s+block from defend cards\b",
        sentence,
    )
    if block_bonus_match:
        return {
            "special": "defend_block_bonus",
            "payload": {
                "target": "self",
                "defend_block_bonus": int(block_bonus_match.group(1)),
            },
        }
    if "first time you gain block from a card each turn" in sentence and "double" in sentence:
        return {
            "special": "first_card_block_double",
            "payload": {"target": "self", "first_card_block_double": 1},
        }
    if (
        "first attack or skill you play each turn" in sentence
        and "top of your draw pile" in sentence
    ):
        return {
            "special": "first_attack_skill_to_draw_top",
            "payload": {"target": "self", "first_attack_skill_to_draw_top": 1},
        }
    if "poison is triggered" in sentence and "additional time" in sentence:
        amount = _first_int(sentence) or 1
        return {
            "special": "poison_extra_triggers",
            "payload": {"target": "self", "poison_extra_triggers": amount},
        }
    return None


def _combat_trigger_payload(sentence: str, *, card_id: str = "") -> Mapping[str, Any] | None:
    turn_start_payload = _turn_start_trigger_payload(sentence)
    if turn_start_payload is not None:
        return turn_start_payload

    turn_end_payload = _turn_end_trigger_payload(sentence)
    if turn_end_payload is not None:
        return turn_end_payload

    combat_end_match = re.search(r"\bat the end of combat, gain\s+(\d+)\s+gold\b", sentence)
    if combat_end_match:
        return {
            "trigger": "combat_end",
            "duration": "combat",
            "effects": ({"gold": int(combat_end_match.group(1))},),
            "text": sentence,
        }
    if (
        sentence.startswith("at the end of combat")
        and "may remove a card" in sentence
        and "deck" in sentence
    ):
        return {
            "trigger": "combat_end",
            "duration": "combat",
            "effects": ({"optional_deck_remove": 1},),
            "text": sentence,
        }

    exhausted_payload = _card_exhausted_trigger_payload(sentence, card_id=card_id)
    if exhausted_payload is not None:
        return exhausted_payload

    block_gain_payload = _player_block_gain_trigger_payload(sentence)
    if block_gain_payload is not None:
        return block_gain_payload

    card_played_payload = _card_played_trigger_payload(sentence)
    if card_played_payload is not None:
        return card_played_payload

    card_drawn_payload = _card_drawn_trigger_payload(sentence)
    if card_drawn_payload is not None:
        return card_drawn_payload

    return None


def _turn_start_trigger_payload(sentence: str) -> Mapping[str, Any] | None:
    if not sentence.startswith("at the start") or "your turn" not in sentence:
        return None

    if "random attack from your discard pile" in sentence and "hand" in sentence:
        return {
            "trigger": "turn_start",
            "duration": "combat",
            "effects": (
                {
                    "move_random_card": {
                        "from": "discard_pile",
                        "destination": "hand",
                        "card_type": "attack",
                        "upgrade": "upgrade" in sentence,
                    }
                },
            ),
            "text": sentence,
        }

    poison_match = re.search(r"\bapply\s+(\d+)\s+poison\b", sentence)
    if poison_match and "all enemies" in sentence:
        return {
            "trigger": "turn_start",
            "duration": "combat",
            "effects": (
                {
                    "apply_status": {
                        "target": "all_enemies",
                        "poison": int(poison_match.group(1)),
                    }
                },
            ),
            "text": sentence,
        }

    random_card = _random_card_to_hand_payload(sentence)
    if random_card is not None:
        return {
            "trigger": "turn_start",
            "duration": "combat",
            "effects": ({"add_random_card_to_hand": random_card},),
            "text": sentence,
        }

    if "play the top card of your draw pile" in sentence:
        return {
            "trigger": "turn_start",
            "duration": "combat",
            "effects": ({"play_top_card": {"zone": "draw_pile"}},),
            "text": sentence,
        }

    if "trigger the passive ability of your rightmost orb" in sentence:
        return {
            "trigger": "turn_start",
            "duration": "combat",
            "effects": ({"trigger_orb_passive": {"selector": "rightmost"}},),
            "text": sentence,
        }

    return None


def _turn_end_trigger_payload(sentence: str) -> Mapping[str, Any] | None:
    if not sentence.startswith("at the end"):
        return None
    retain_match = re.search(r"\bretain up to\s+(\d+)\s+cards?\b", sentence)
    if retain_match:
        return {
            "trigger": "turn_end",
            "duration": "combat",
            "effects": (
                {
                    "choose_card": {
                        "action": "add_retain",
                        "zone": "hand",
                        "count": int(retain_match.group(1)),
                        "text": sentence,
                    }
                },
            ),
            "text": sentence,
        }
    if "random attack in your hand" in sentence and "played against a random enemy" in sentence:
        return {
            "trigger": "turn_end",
            "duration": "combat",
            "effects": (
                {
                    "play_random_card_from_hand": {
                        "card_type": "attack",
                        "target": "random_enemy",
                    }
                },
            ),
            "text": sentence,
        }
    delayed_damage = re.search(r"\bat the end of\s+(\d+)\s+turns?, deal\s+(\d+)\s+damage", sentence)
    if delayed_damage and "all enemies" in sentence:
        return {
            "trigger": "turn_end",
            "duration": "combat",
            "delay": int(delayed_damage.group(1)),
            "effects": ({"all_damage": int(delayed_damage.group(2))},),
            "text": sentence,
        }
    return None


def _card_exhausted_trigger_payload(
    sentence: str,
    *,
    card_id: str = "",
) -> Mapping[str, Any] | None:
    self_exhaust = any(
        phrase in sentence
        for phrase in (
            "when this card is exhausted",
            "when this card is [gold]exhausted",
            "when this is exhausted",
            "when this is [gold]exhausted",
        )
    )
    if "whenever a card is exhausted" not in sentence and not self_exhaust:
        return None
    effects: list[Mapping[str, Any]] = []
    block_match = re.search(r"\bgain\s+(\d+)\s+block\b", sentence)
    if block_match:
        effects.append({"block": int(block_match.group(1))})
    draw_match = re.search(r"\bdraw\s+(\d+)\s+card", sentence)
    if draw_match:
        effects.append({"draw": int(draw_match.group(1))})
    energy_match = re.search(r"\bgain\s+\[energy:(\d+)\]", sentence)
    if energy_match:
        effects.append({"energy": int(energy_match.group(1))})
    inline_energy_match = re.search(r"\bgain\s+(\d+)\s+\[energy:\d+\]", sentence)
    if inline_energy_match:
        effects.append({"energy": int(inline_energy_match.group(1))})
    if not effects:
        return None
    payload: dict[str, Any] = {
        "trigger": "card_exhausted",
        "duration": "once" if self_exhaust else "combat",
        "effects": tuple(effects),
        "text": sentence,
    }
    if self_exhaust and card_id:
        payload["condition"] = {"card_id": card_id}
    return payload


def _player_block_gain_trigger_payload(sentence: str) -> Mapping[str, Any] | None:
    if "whenever you gain block on your turn" not in sentence:
        return None
    if "half that much block" not in sentence:
        return None
    return {
        "trigger": "player_block_gain",
        "duration": "combat",
        "condition": {"from_card": True},
        "effects": (
            {"ally_block": {"formula": "half_context_amount", "source": "beacon_of_hope"}},
        ),
        "text": sentence,
    }


def _card_played_trigger_payload(sentence: str) -> Mapping[str, Any] | None:
    if not any(
        phrase in sentence
        for phrase in (
            "whenever you play",
            "every time you play",
            "when you play",
            "attack you play each turn",
        )
    ):
        return None

    if "third attack you play each turn" in sentence and "copy" in sentence:
        return {
            "trigger": "card_played",
            "duration": "combat",
            "condition": {"card_type": "attack"},
            "counter_scope": "turn",
            "every": 3,
            "effects": ({"copy_context_card_to_hand": {"context": "played_card"}},),
            "text": sentence,
        }

    if "when you play a skill" in sentence and "gains sly" in sentence:
        return {
            "trigger": "card_played",
            "duration": "combat",
            "condition": {"card_type": "skill"},
            "effects": (
                {
                    "add_keyword_to_context_card": {
                        "context": "played_card",
                        "keyword": "sly",
                        "amount": 1,
                    }
                },
            ),
            "text": sentence,
        }

    if "every time you play" in sentence and "cards in a single turn" in sentence:
        every_match = re.search(r"\bevery time you play\s+(\d+)\s+cards", sentence)
        damage_match = re.search(r"\bdeal\s+(\d+)\s+damage\b", sentence)
        if every_match and damage_match and "all enemies" in sentence:
            return {
                "trigger": "card_played",
                "duration": "combat",
                "counter_scope": "turn",
                "every": int(every_match.group(1)),
                "effects": ({"all_damage": int(damage_match.group(1))},),
                "text": sentence,
            }

    condition: dict[str, Any] = {}
    if "attack" in sentence:
        condition["card_type"] = "attack"
    elif "skill" in sentence:
        condition["card_type"] = "skill"
    if "ethereal" in sentence:
        condition["keyword"] = "ethereal"

    block_match = re.search(r"\bgain\s+(\d+)\s+block\b", sentence)
    if block_match:
        return {
            "trigger": "card_played",
            "duration": "turn" if "this turn" in sentence else "combat",
            "condition": condition,
            "effects": ({"block": int(block_match.group(1))},),
            "text": sentence,
        }

    strength_match = re.search(r"\bgain\s+(\d+)\s+strength\b", sentence)
    if strength_match and "this turn" in sentence:
        return {
            "trigger": "card_played",
            "duration": "turn",
            "condition": condition,
            "effects": (
                {
                    "apply_status": {
                        "target": "self",
                        "temporary_strength": int(strength_match.group(1)),
                    }
                },
            ),
            "text": sentence,
        }

    random_card = _random_card_to_hand_payload(sentence)
    if random_card is not None:
        return {
            "trigger": "card_played",
            "duration": "combat",
            "condition": condition,
            "effects": ({"add_random_card_to_hand": random_card},),
            "text": sentence,
        }

    return None


def _card_drawn_trigger_payload(sentence: str) -> Mapping[str, Any] | None:
    if "whenever you draw a card" not in sentence:
        return None
    if (
        "containing" in sentence
        and "strike" in sentence
        and "played against a random enemy" in sentence
    ):
        return {
            "trigger": "card_drawn",
            "duration": "combat",
            "condition": {"name_contains": "strike"},
            "effects": (
                {
                    "play_context_card": {
                        "context": "drawn_card",
                        "from": "hand",
                        "target": "random_enemy",
                    }
                },
            ),
            "text": sentence,
        }
    poison_match = re.search(r"\bapply\s+(\d+)\s+poison\b", sentence)
    if poison_match and "all enemies" in sentence:
        return {
            "trigger": "card_drawn",
            "duration": "turn" if "this turn" in sentence else "combat",
            "effects": (
                {
                    "apply_status": {
                        "target": "all_enemies",
                        "poison": int(poison_match.group(1)),
                    }
                },
            ),
            "text": sentence,
        }
    return None


def _random_card_to_hand_payload(sentence: str) -> Mapping[str, Any] | None:
    if "random" not in sentence or "hand" not in sentence:
        return None
    if not ("add" in sentence or "adds" in sentence):
        return None

    payload: dict[str, Any] = {"count": 1, "pool": "character"}
    rarity_match = re.search(r"\brandom\s+(common|uncommon|rare)\s+", sentence)
    if rarity_match:
        payload["rarity"] = rarity_match.group(1)
    for word, card_type in (
        ("attack", "attack"),
        ("skill", "skill"),
        ("power", "power"),
    ):
        if re.search(rf"\b{word}\b", sentence):
            payload["card_types"] = (card_type,)
            break
    if "colorless" in sentence:
        payload["pool"] = "colorless"
    return payload


def _first_int(sentence: str) -> int | None:
    match = re.search(r"\b(\d+)\b", sentence)
    return int(match.group(1)) if match else None


def _last_int(sentence: str) -> int | None:
    matches = re.findall(r"\b(\d+)\b", sentence)
    return int(matches[-1]) if matches else None


def _first_int_after(sentence: str, marker: str) -> int | None:
    index = sentence.find(marker)
    if index < 0:
        return None
    return _first_int(sentence[index + len(marker) :])


def _zone_from_sentence(sentence: str) -> str | None:
    for phrase, zone in (
        ("draw pile", "draw_pile"),
        ("discard pile", "discard_pile"),
        ("exhaust pile", "exhaust_pile"),
        ("hand", "hand"),
        ("deck", "deck"),
    ):
        if phrase in sentence:
            return zone
    return None


def _choice_source_zone(sentence: str) -> str | None:
    match = re.search(
        r"\b(?:from|in|of)\s+your\s+(?P<zone>hand|draw pile|discard pile|exhaust pile|deck)\b",
        sentence,
    )
    if match:
        return _zone_phrase_to_key(match.group("zone"))
    if "add retain" in sentence and "card in your hand" in sentence:
        return "hand"
    return _zone_from_sentence(sentence)


def _zone_phrase_to_key(phrase: str) -> str:
    return phrase.strip().lower().replace(" ", "_")


def _choice_count(sentence: str) -> int | None:
    match = re.search(r"\bchoose\s+(\d+|a|an|one)\b", sentence)
    if match:
        return _word_amount(match.group(1))
    match = re.search(r"\b(?:transform|exhaust|put|add retain to)\s+(\d+|a|an|one)\b", sentence)
    if match:
        return _word_amount(match.group(1))
    return None


def _choice_pool_count(sentence: str) -> int | None:
    match = re.search(r"\bchoose\s+\d+\s+of\s+(\d+)\b", sentence)
    return int(match.group(1)) if match else None


def _choice_action(sentence: str) -> str:
    if "transform" in sentence:
        return "transform"
    if re.search(r"\bdiscard\s+(?:\d+|a|an|one)?\s*cards?\b", sentence):
        return "discard"
    if "play it" in sentence or "play them" in sentence:
        return "play"
    if "copy" in sentence or "copies" in sentence:
        return "copy"
    if "add retain" in sentence:
        return "add_retain"
    if "exhaust" in sentence:
        return "exhaust"
    if "top of your draw pile" in sentence:
        return "move_to_draw_top"
    if "put" in sentence or "add into your hand" in sentence:
        return "move_to_hand"
    return "select"


def _generated_choice_pool(sentence: str) -> str:
    if "colorless" in sentence:
        return "colorless"
    if "another character" in sentence:
        return "other_character"
    return "any"


def _choice_is_copy_to_hand(sentence: str) -> bool:
    return (
        re.search(r"\bchoose\s+an?\s+(?:attack|power|colorless)", sentence) is not None
        or "add a copy" in sentence
    )


def _choice_is_next_turn_copy(sentence: str, context: str) -> bool:
    return (
        sentence == "choose a card"
        and re.search(r"\bnext turn,?\s+add\s+\d+\s+copies\b", context) is not None
        and "into your hand" in context
    )


def _next_turn_copy_count(context: str) -> int:
    match = re.search(r"\bnext turn,?\s+add\s+(\d+)\s+copies\b", context)
    return int(match.group(1)) if match else 1


def _draw_amount_from_sentence(sentence: str) -> int:
    match = re.search(r"\bdraw\s+(\d+|a|an|one)\b", sentence)
    return _word_amount(match.group(1)) if match else 0


def _choice_play_times(sentence: str) -> int:
    match = re.search(r"\bplay\s+(?:it|them)\s+(\d+)\s+times\b", sentence)
    if match:
        return int(match.group(1))
    if "play it" in sentence or "play them" in sentence:
        return 1
    return 1


def _transform_target_card_id(sentence: str) -> str | None:
    match = re.search(r"\binto\s+(?P<target>[a-z0-9' ,_-]+?)(?:\.|$)", sentence)
    if not match:
        return None
    target = match.group("target").strip()
    target = re.sub(r"\s+", " ", target)
    target = target.removesuffix("s") if target.endswith(" bombs") else target
    return _normalized_id(target) or None


def _choice_card_types(sentence: str) -> tuple[str, ...]:
    types: list[str] = []
    for word, card_type in (
        ("attack", "attack"),
        ("skill", "skill"),
        ("power", "power"),
    ):
        if re.search(rf"\b{word}s?\b", sentence):
            types.append(card_type)
    return tuple(types)


def _requires_implicit_card_choice(sentence: str) -> bool:
    return bool(
        re.search(
            r"\b(transform|exhaust)\s+(?:\d+|a|an|one)?\s*cards?\s+"
            r"(?:in|from)\s+your\s+(?:hand|draw pile|discard pile|deck)",
            sentence,
        )
        or re.search(
            r"\bdiscard\s+(?:\d+|a|an|one)?\s*cards?\s+"
            r"(?:in|from)\s+your\s+(?:hand|draw pile|discard pile|deck)",
            sentence,
        )
        or re.search(
            r"\bput\s+(?:a|an|one|\d+)\s+(?:skill|attack|power|card)\s+"
            r"from\s+your\s+(?:hand|draw pile|discard pile|deck)",
            sentence,
        )
        or re.search(r"\badd retain to\s+(?:a|an|one|\d+)\s+card\s+in your hand", sentence)
    )


def _word_amount(raw: str) -> int:
    if raw in {"a", "an", "one"}:
        return 1
    return int(raw)


def _payload_amount(payload: Any) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    amount = payload.get("amount")
    if isinstance(amount, bool):
        return int(amount)
    if isinstance(amount, int):
        return amount
    return None


def _has_any_word(sentence: str, words: Sequence[str]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", sentence) for word in words)


def _plan_status(
    steps: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, Any]],
) -> CardSpecialStatus:
    if steps and blockers:
        return "partial"
    if blockers:
        return "explicit_blocker"
    if steps:
        return "executable"
    return "none"


def _blocker_reasons(blockers: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    reasons: list[str] = []
    for blocker in blockers:
        payload = blocker.get("explicit_blocker")
        if not isinstance(payload, Mapping):
            continue
        reason = payload.get("reason")
        if isinstance(reason, str) and reason not in reasons:
            reasons.append(reason)
    return tuple(reasons)


def _card_id(card_spec: Mapping[str, Any]) -> str:
    return _normalized_id(
        card_spec.get("card_id", card_spec.get("id", card_spec.get("name", "unknown_card")))
    )


def _clone_jsonish_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _clone_jsonish(value) for key, value in source.items()}


def _clone_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _clone_jsonish_mapping(value)
    if isinstance(value, tuple):
        return tuple(_clone_jsonish(item) for item in value)
    if isinstance(value, list):
        return [_clone_jsonish(item) for item in value]
    return value


def _normalized_id(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("'", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


__all__ = [
    "CARD_SPECIAL_EFFECT_KEYS",
    "CardSpecialPlan",
    "CardSpecialStatus",
    "card_special_blockers",
    "card_special_events",
    "card_special_plan",
    "normalize_card_special_steps",
]

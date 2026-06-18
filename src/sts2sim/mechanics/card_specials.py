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
        "apply_status",
        "channel_orb",
        "evoke_orb",
        "explicit_blocker",
        "orb_slot_delta",
        "player_resource",
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
    sentences = _sentences(_description_from(card_spec))
    steps: list[Mapping[str, Any]] = []
    blockers: list[Mapping[str, Any]] = []
    events: list[Mapping[str, Any]] = []

    for sentence in sentences:
        _collect_if_kill_resource(sentence, card_id=card_id, blockers=blockers, events=events)
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
        _collect_choice_blockers(sentence, card_id=card_id, blockers=blockers, events=events)

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


def _collect_if_kill_resource(
    sentence: str,
    *,
    card_id: str,
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    for resource, amount in re.findall(
        r"\bif\b[^.]*kills? an enemy[^.]*gain\s+\[(energy|star):(\d+)\]",
        sentence,
    ):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="if_kill_resource_gain",
            reason="requires post-damage kill detection before granting the resource",
            metadata={
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
        if _is_timed_or_triggered(sentence):
            _append_blocker(
                blockers,
                events,
                card_id=card_id,
                kind="timed_or_triggered_orb_effect",
                reason="requires a combat timing hook before channeling orbs",
                metadata={"text": sentence},
            )
        elif _has_dynamic_orb_channel_count(sentence):
            _append_blocker(
                blockers,
                events,
                card_id=card_id,
                kind="dynamic_orb_channel_count",
                reason="requires combat state to determine how many orbs to channel",
                metadata={"text": sentence},
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
        if _is_timed_or_triggered(sentence):
            _append_blocker(
                blockers,
                events,
                card_id=card_id,
                kind="timed_or_triggered_orb_effect",
                reason="requires a combat timing hook before evoking orbs",
                metadata={"text": sentence},
            )
        elif "evoke all of your orbs" in sentence:
            _append_step(
                steps,
                events,
                card_id=card_id,
                special="evoke_orb",
                effect_key="evoke_orb",
                payload={"selector": "all", "amount": "all"},
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
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="orb_state_scaling",
            reason="requires current orb state to scale the card effect",
            metadata={"text": sentence},
        )
    if "trigger the passive ability" in sentence and "orb" in sentence:
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
    if _is_timed_or_triggered(sentence):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="timed_or_triggered_forge",
            reason="requires a combat timing hook before granting forge",
            metadata={"text": sentence},
        )
        return
    if re.search(r"\bforge\s+x\b", sentence) or "for every other time" in sentence:
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="dynamic_forge_amount",
            reason="requires combat state or X-cost energy spent to determine forge",
            metadata={"text": sentence},
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
    if re.search(r"\bsummon\s+\d+\s+x\b", sentence) or re.search(r"\bsummon\s+x\b", sentence):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="dynamic_summon_amount",
            reason="requires X-cost energy spent to determine summon",
            metadata={"text": sentence},
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
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="osty_state_required",
            reason="requires Osty companion state before resolving the effect",
            metadata={"text": sentence},
        )
        return
    if "additional damage equal" in sentence or "for all your other osty attacks" in sentence:
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="dynamic_osty_damage",
            reason="requires Osty state or combat history to determine damage",
            metadata={"text": sentence},
        )

    for match in re.finditer(
        r"\bosty(?:'s)?\s+deals\s+(\d+)\s+damage(?:\s+to\s+(all enemies|a random enemy))?",
        sentence,
    ):
        target = _osty_target(match.group(2) or "")
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="osty_action_required",
            reason="requires Osty companion state before resolving the action",
            metadata={"action": "damage", "amount": int(match.group(1)), "target": target},
        )

    heal_match = re.search(r"\bosty(?:'s)?\s+heals\s+(\d+)\s+hp\b", sentence)
    if heal_match:
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="osty_action_required",
            reason="requires Osty companion state before resolving the action",
            metadata={"action": "heal", "amount": int(heal_match.group(1))},
        )

    if re.search(r"\bosty(?:'s)?\s+dies\b", sentence):
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="osty_action_required",
            reason="requires Osty companion state before resolving the action",
            metadata={"action": "die"},
        )

    if "osty's attacks deal" in sentence:
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="osty_persistent_modifier",
            reason="requires a companion modifier that persists beyond this source pass",
            metadata={"text": sentence},
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
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="soul_play_trigger",
            reason="requires a card-play trigger for Soul cards",
            metadata={"text": sentence},
        )
    if "for each soul" in sentence:
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
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="dynamic_soul_creation_count",
            reason="requires X-cost energy spent to determine how many Souls to add",
            metadata={"text": sentence},
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
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="sovereign_blade_state_required",
            reason="requires Sovereign Blade card state before modifying the blade",
            metadata={
                "action": "temporary_modifier",
                "modifier": "double_damage",
                "duration": "turn",
                "target": "enemy",
            },
        )
    if "now deals damage to all enemies" in sentence:
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="sovereign_blade_state_required",
            reason="requires Sovereign Blade card state before modifying the blade",
            metadata={"action": "set_target", "target": "all_enemies"},
        )
    if "now hits an additional time" in sentence:
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="sovereign_blade_state_required",
            reason="requires Sovereign Blade card state before modifying the blade",
            metadata={"action": "add_hit", "amount": 1},
        )
    if "put sovereign blade into your hand from anywhere" in sentence:
        _append_blocker(
            blockers,
            events,
            card_id=card_id,
            kind="sovereign_blade_state_required",
            reason="requires Sovereign Blade card state before moving the blade",
            metadata={"action": "move_to_hand", "source": "anywhere"},
        )
    if "sovereign blade" in sentence and not any(
        phrase in sentence
        for phrase in (
            "deals double damage",
            "now deals damage to all enemies",
            "now hits an additional time",
            "put sovereign blade into your hand from anywhere",
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


def _collect_choice_blockers(
    sentence: str,
    *,
    card_id: str,
    blockers: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    if "choose" in sentence:
        discovery = "random" in sentence and _zone_from_sentence(sentence) is None
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
                "zone": _zone_from_sentence(sentence),
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
                "zone": _zone_from_sentence(sentence),
                "count": _choice_count(sentence),
                "action": _choice_action(sentence),
            },
        )


def _append_resource_step(
    steps: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
    *,
    card_id: str,
    resource: str,
    amount: int,
    special: str,
) -> None:
    _append_step(
        steps,
        events,
        card_id=card_id,
        special=special,
        effect_key="player_resource",
        payload={"resource": resource, "amount": amount, "source": "card_special"},
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
    if "play it" in sentence or "play them" in sentence:
        return "play"
    if "copy" in sentence or "copies" in sentence:
        return "copy"
    if "add retain" in sentence:
        return "add_retain"
    if "exhaust" in sentence:
        return "exhaust"
    if "put" in sentence or "add into your hand" in sentence:
        return "move_to_hand"
    return "select"


def _requires_implicit_card_choice(sentence: str) -> bool:
    return bool(
        re.search(
            r"\b(transform|exhaust)\s+(?:\d+|a|an|one)?\s*cards?\s+"
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

"""Validated stable starter defaults when an explicit character source is absent.

Values match the English Spire Codex v0.107.1 stable character/card snapshot.
The engine prefers local cached source rows and caller overrides when available.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_CHARACTERS = {
    "ironclad": (80, "burning_blood", 5, 4, ("bash",)),
    "silent": (70, "ring_of_the_snake", 5, 5, ("neutralize", "survivor")),
    "regent": (75, "divine_right", 4, 4, ("falling_star", "venerate")),
    "necrobinder": (66, "bound_phylactery", 4, 4, ("bodyguard", "unleash")),
    "defect": (75, "cracked_core", 4, 4, ("zap", "dualcast")),
}
_SPECIALS: dict[str, tuple[str, int, dict[str, Any]]] = {
    "bash": ("attack", 2, {"damage": 8, "apply_status": {"target": "enemy", "vulnerable": 2}}),
    "neutralize": ("attack", 0, {"damage": 3, "apply_status": {"target": "enemy", "weak": 1}}),
    "survivor": ("skill", 1, {"block": 8, "discard_choice": 1}),
    "falling_star": (
        "attack",
        0,
        {
            "damage": 8,
            "apply_status": {"target": "enemy", "weak": 1, "vulnerable": 1},
        },
    ),
    "venerate": ("skill", 1, {"player_resource": {"resource": "star", "amount": 2}}),
    "bodyguard": ("skill", 1, {"player_resource": {"resource": "summon", "amount": 5}}),
    "unleash": ("attack", 1, {"osty_action": {"action": "damage", "amount": 6, "target": "enemy"}}),
    "zap": ("skill", 1, {"channel_orb": {"orb": "lightning", "amount": 1}}),
    "dualcast": ("skill", 1, {"evoke_orb": {"selector": "rightmost", "amount": 2}}),
}


def character_starter_defaults(character_id: str) -> dict[str, Any]:
    character = character_id.lower()
    values = _CHARACTERS.get(character)
    if values is None:
        return {}
    hp, relic, strikes, defends, specials = values
    return {
        "id": character.upper(),
        "starting_hp": hp,
        "starting_gold": 99,
        "max_energy": 3,
        "starting_relics": (relic,),
        "starting_deck": (
            (f"strike_{character}",) * strikes + (f"defend_{character}",) * defends + specials
        ),
        "orb_slots": 3 if character == "defect" else None,
    }


def starter_card_fallbacks(character_id: str) -> dict[str, dict[str, Any]]:
    character = character_id.lower()
    if character not in _CHARACTERS:
        return {}
    specs: dict[str, dict[str, Any]] = {}
    for name, card_type, effect in (
        ("strike", "attack", {"damage": 6}),
        ("defend", "skill", {"block": 5}),
    ):
        card_id = f"{name}_{character}"
        specs[card_id] = {
            "card_id": card_id,
            "name": name.title(),
            "type": card_type,
            "cost": 1,
            "target": "enemy" if card_type == "attack" else "self",
            "effects": effect,
        }
    for card_id in _CHARACTERS[character][-1]:
        card_type, cost, effects = _SPECIALS[card_id]
        specs[card_id] = {
            "card_id": card_id,
            "name": card_id.replace("_", " ").title(),
            "type": card_type,
            "cost": cost,
            "target": "enemy" if card_type == "attack" else "self",
            "effects": deepcopy(effects),
        }
    return specs

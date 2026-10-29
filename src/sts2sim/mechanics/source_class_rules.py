"""Independent stable-source rules for the orb, Osty, star and auxiliary cards.

The public card catalog contains dynamic variables that are not immediate
effects (notably ``Cards`` on transforms and generated-card effects).  These
rules describe the actual OnPlay order instead of interpreting those variables
as draws. See tests/fixtures/source_class_rules.json for source references.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def source_class_spec(source: Mapping[str, Any]) -> Mapping[str, Any]:
    """Tyranny's Exhaust hover tip describes its choice, not a card keyword."""
    if str(source.get("id", source.get("card_id", ""))).lower() != "tyranny":
        return source
    result = dict(source)
    for key in ("keywords", "keywords_key", "tags"):
        values = source.get(key)
        if isinstance(values, (list, tuple)):
            result[key] = tuple(value for value in values if str(value).lower() != "exhaust")
    result["exhausts"] = False
    return result


def source_class_generation_steps(
    source: Mapping[str, Any],
    steps: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    """Canonical CardFactory.GetDistinctForCombat calls select without replacement."""
    card_id = str(source.get("id", source.get("card_id", ""))).lower()
    if card_id not in {
        "bundle_of_joy",
        "manifest_authority",
        "metamorphosis",
        "mad_science",
        "white_noise",
        "distraction",
        "largesse",
        "creative_ai",
        "call_of_the_void",
        "spectrum_shift",
        "hello_world",
    }:
        return steps

    def tagged(value: Any) -> Any:
        if isinstance(value, Mapping):
            result = {key: tagged(item) for key, item in value.items()}
            if "add_random_card_to_hand" in result:
                payload = result["add_random_card_to_hand"]
                result["add_random_card_to_hand"] = {
                    **(dict(payload) if isinstance(payload, Mapping) else {"count": int(payload)}),
                    "distinct": True,
                }
            if result.get("operation") == "random_generate":
                result["distinct"] = True
            return result
        if isinstance(value, (list, tuple)):
            return tuple(tagged(item) for item in value)
        return value

    return tuple(tagged(step) for step in steps)


def source_class_steps(
    source: Mapping[str, Any],
    steps: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    if source.get("effects"):
        return steps
    card_id = str(source.get("id", source.get("card_id", ""))).lower()
    upgraded = bool(source.get("upgraded"))
    text = re.sub(r"\[/?gold\]", "", str(source.get("description", ""))).lower()

    def number(pattern: str, default: int = 0) -> int:
        match = re.search(pattern, text)
        return int(match[1]) if match else default

    def marker(operation: str, **values: Any) -> Mapping[str, Any]:
        return {"source_class": {"operation": operation, **values}}

    def status(name: str, amount: int, target: str = "self") -> Mapping[str, Any]:
        return {"apply_status": {"target": target, name: amount}}

    def choice(action: str, zone: str, count: int = 1, **values: Any) -> Mapping[str, Any]:
        return {"choose_card": {"action": action, "zone": zone, "count": count, **values}}

    def resource(name: str, amount: Any) -> Mapping[str, Any]:
        return {"player_resource": {"resource": name, "amount": amount}}

    if card_id == "consuming_shadow":
        return ({"channel_orb": {"orb": "dark", "amount": 3 if upgraded else 2}},
                status("consuming_shadow", 1))

    def generated(
        card: str, destination: str, count: Any = 1, *, upgrade: bool = False
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "card_id": card,
            "destination": destination,
            "upgraded": upgrade,
            "position": "random",
        }
        payload["count_formula" if isinstance(count, Mapping) else "count"] = count
        return marker("generate", **payload)

    damage = number(r"deal (\d+) damage", int(source.get("damage") or 0))
    block = number(r"gain (\d+) block", int(source.get("block") or 0))
    draw = number(r"draw (\d+) cards?", int(source.get("cards_draw") or 0))

    # Explicit source ordering also matters with an orb evocation, artifact,
    # or a damage-on-debuff power between these operations.
    if card_id in {"falling_star", "gamma_blast", "comet", "meteor_shower"}:
        all_enemies = card_id == "meteor_shower"
        target = "all_enemies" if all_enemies else "enemy"
        amount = number(r"apply (\d+) weak")
        return (
            {"all_damage" if all_enemies else "damage": damage},
            status("weak", amount, target),
            status("vulnerable", amount, target),
        )
    if card_id == "coolheaded":
        return ({"channel_orb": {"orb": "frost", "amount": 1}}, {"draw": draw})
    if card_id in {"focused_strike", "hotfix", "hyperbeam", "feeding_frenzy"}:
        stat = (
            "temporary_strength"
            if card_id == "feeding_frenzy"
            else ("focus" if card_id == "hyperbeam" else "temporary_focus")
        )
        amount = number(r"(?:gain|lose) (\d+) (?:focus|strength)")
        benefit = status(stat, -amount if card_id == "hyperbeam" else amount)
        return (
            ({"all_damage" if card_id == "hyperbeam" else "damage": damage},) if damage else ()
        ) + (benefit,)
    if card_id in {"crush_under", "dying_star"}:
        return (
            {"all_damage": damage},
            status("temporary_strength", -number(r"lose (\d+) strength"), "all_enemies"),
        )
    if card_id == "resonance":
        return (
            status("strength", number(r"gain (\d+) strength")),
            status("strength", -1, "all_enemies"),
        )
    if card_id == "reflect":
        return ({"block": block}, status("reflect", 1))
    if card_id == "parry":
        return (marker("blade_power", power="parry", amount=number(r"gains (\d+) block", 10)),)
    if card_id == "seeking_edge" and "forge" in text:
        return (
            marker("blade_power", power="seeking_edge", amount=1),
            resource("forge", number(r"forge (\d+)", 7)),
        )
    if card_id == "sword_sage" and "replay" in text:
        return (marker("blade_power", power="sword_sage", amount=number(r"replay (\d+)", 1)),)
    if card_id == "shatter":
        return ({"all_damage": damage}, marker("shatter"))
    if card_id == "tesla_coil":
        return ({"damage": damage}, marker("target_lightning", repeats=2 if upgraded else 1))
    if card_id == "whistle":
        return ({"damage": damage}, marker("stun"))
    if card_id == "stack":
        return ({"block_formula": {"formula": "discard_pile_count", "base": 3 if upgraded else 0}},)
    if card_id == "conqueror" and "forge" in text:
        return (resource("forge", number(r"forge (\d+)", 3)), status("conqueror", 1, "enemy"))

    if card_id in {"charge_battery", "delay", "hegemony", "refine_blade", "outmaneuver"}:
        first = tuple(
            step
            for step in steps
            if any(key in step for key in ("damage", "block", "player_resource"))
        )
        return first + ({"next_turn": {"energy": number(r"gain \[energy:(\d+)\]")}},)
    if card_id == "invoke":
        return (
            {
                "combat_trigger": {
                    "trigger": "turn_start",
                    "duration": "once",
                    "effects": (
                        resource("summon", number(r"summon (\d+)")),
                        {"energy": number(r"gain \[energy:(\d+)\]")},
                    ),
                }
            },
        )
    if card_id == "relax":
        return (
            {"block": block},
            {"next_turn": {"energy": number(r"gain \[energy:(\d+)\]"), "draw": draw}},
        )
    if card_id == "scavenge":
        return (
            choice("exhaust", "hand"),
            {"next_turn": {"energy": number(r"gain \[energy:(\d+)\]")}},
        )
    if card_id == "foregone_conclusion":
        return (
            {
                "timed_choice": {
                    "trigger": "turn_start",
                    "repeat": False,
                    "choose_card": {
                        "action": "move_to_hand",
                        "zone": "draw_pile",
                        "count": number(r"put (\d+) cards?", 2),
                    },
                }
            },
        )
    if card_id == "hidden_cache":
        return (
            resource("star", number(r"gain \[star:(\d+)\]")),
            {"next_turn": {"star": number(r"next turn,?\s*gain \[star:(\d+)\]")}},
        )

    if card_id in {"begone", "charge", "guards", "seance"}:
        target, zone, count = {
            "begone": ("minion_strike", "hand", 1),
            "charge": ("minion_dive_bomb", "draw_pile", 2),
            "guards": ("minion_sacrifice", "hand", 99),
            "seance": ("soul", "draw_pile", 1),
        }[card_id]
        return (
            choice(
                "transform",
                zone,
                count,
                target_card_id=target,
                optional=card_id == "guards",
                upgrade_result=upgraded and card_id != "seance",
            ),
        )
    if card_id == "compact":
        return (
            {"block": block},
            marker("transform_statuses", target_card_id="fuel", upgraded=upgraded),
        )
    if card_id in {"dredge", "neows_fury", "dual_wield"}:
        if card_id == "dual_wield":
            return tuple(step for step in steps if "choose_card" in step)
        count = number(r"put (?:up to )?(\d+) cards?", 3)
        return (({"damage": damage},) if damage else ()) + (
            choice(
                "move_to_hand",
                "discard_pile",
                count,
                optional=card_id == "neows_fury",
                hand_space_limit=True,
            ),
        )
    if card_id == "anointed":
        return (marker("retrieve", zone="draw_pile", rarity="rare", random=True),)
    if card_id == "all_for_one":
        return ({"damage": damage}, marker("retrieve", zone="discard_pile", zero_cost=True))
    if card_id in {"bundle_of_joy", "manifest_authority"}:
        amount = number(r"add (\d+) random", 1)
        return (({"block": block},) if block else ()) + (
            {"add_random_card_to_hand": {"pool": "colorless", "count": amount}},
        )
    if card_id == "quasar":
        return (
            choice(
                "move_to_hand",
                "generated",
                choices=3,
                pool="colorless",
                optional=True,
                upgraded=upgraded,
            ),
        )
    if card_id == "metamorphosis":
        return (
            marker(
                "random_generate",
                count=number(r"add (\d+) random", 3),
                card_types=("attack",),
                destination="draw_pile",
                free_to_play_this_combat=True,
            ),
        )

    if card_id in {
        "reave",
        "grave_warden",
        "capture_spirit",
        "glimpse_beyond",
        "dirge",
        "severance",
    }:
        prefix: tuple[Mapping[str, Any], ...] = ()
        if card_id == "capture_spirit":
            prefix = (marker("target_hp_loss", amount=number(r"loses (\d+) hp")),)
        elif damage:
            prefix = ({"damage": damage},)
        elif block:
            prefix = ({"block": block},)
        if card_id == "severance":
            return prefix + tuple(
                generated("soul", zone) for zone in ("draw_pile", "discard_pile", "hand")
            )
        if card_id == "dirge":
            return (marker("dirge", summon=number(r"summon (\d+)", 3), upgraded=upgraded),)
        count = number(r"add (\d+) souls?", 1)
        return prefix + (
            generated("soul", "draw_pile", count, upgrade=upgraded and card_id == "reave"),
        )
    if card_id == "soul_storm":
        return tuple(step for step in steps if "damage_formula" in step)
    if card_id in {
        "collision_course",
        "boost_away",
        "fight_through",
        "gunk_up",
        "overclock",
        "turbo",
        "crash_landing",
    }:
        token = {
            "collision_course": "debris",
            "crash_landing": "debris",
            "boost_away": "dazed",
            "fight_through": "wound",
            "gunk_up": "slimed",
            "overclock": "burn",
            "turbo": "void",
        }[card_id]
        prefix = tuple(
            step
            for step in steps
            if any(key in step for key in ("damage", "all_damage", "draw", "energy", "block"))
        )
        generated_count: Any = (
            {"formula": "hand_space"} if card_id == "crash_landing" else number(r"add (\d+)", 1)
        )
        return prefix + (
            generated(token, "hand" if token == "debris" else "discard_pile", generated_count),
        )
    if card_id == "energy_surge":
        return (marker("all_players_energy", amount=number(r"gain \[energy:(\d+)\]", 2)),)

    if card_id == "sic_em":
        return (
            {
                "osty_action": {
                    "action": "damage",
                    "target": "enemy",
                    "amount": number(r"deals (\d+) damage"),
                }
            },
            status("sicem", number(r"summon (\d+)"), "enemy"),
        )
    if card_id == "fetch":
        return ({"conditional": {"condition": {"osty_alive": True}, "effects": steps}},)
    if card_id == "bone_shards":
        return (
            {
                "conditional": {
                    "condition": {"osty_alive": True},
                    "effects": (
                        {
                            "osty_action": {
                                "action": "damage",
                                "target": "all_enemies",
                                "amount": number(r"deals (\d+) damage"),
                            }
                        },
                        {"block": block},
                        {"osty_action": {"action": "die"}},
                    ),
                }
            },
        )
    if card_id == "sacrifice":
        return (marker("sacrifice"),)
    if card_id == "high_five":
        return (
            {
                "osty_action": {
                    "action": "damage",
                    "target": "all_enemies",
                    "amount": number(r"deals (\d+) damage"),
                }
            },
            status("vulnerable", number(r"applies (\d+) vulnerable"), "all_enemies"),
        )
    if card_id == "snap":
        return (steps[0], choice("add_retain", "hand", exclude_keyword="retain"))
    if card_id == "sculpting_strike":
        return (
            {"damage": damage},
            choice("add_keyword", "hand", keyword="ethereal", exclude_keyword="ethereal"),
        )
    if card_id == "transfigure":
        return (choice("add_keyword", "hand", keyword="replay", cost_delta=1),)
    if card_id == "no_escape":
        return (marker("no_escape", base=number(r"apply (\d+) doom", 10), extra=5, divisor=10),)
    if card_id == "end_of_days":
        return steps + (marker("doom_kill"),)
    if card_id == "misery":
        return ({"damage": damage}, marker("spread_debuffs"))
    if card_id == "hang":
        return (marker("hang_damage", amount=damage), marker("hang_double"))

    if card_id in {"barrage", "lunar_blast", "pull_from_below"}:
        return (
            marker(
                "repeat_attack",
                amount=damage,
                basis={
                    "barrage": "orbs",
                    "lunar_blast": "skills_turn",
                    "pull_from_below": "ethereal_combat",
                }[card_id],
            ),
        )
    if card_id in {"crescent_spear", "supermassive", "death_march"}:
        return (
            marker(
                "calculated_damage",
                base=damage,
                extra=number(r"deals (\d+) additional damage"),
                basis={
                    "crescent_spear": "star_cards",
                    "supermassive": "created_combat",
                    "death_march": "draws_turn",
                }[card_id],
            ),
        )
    if card_id == "beat_into_shape":
        return (
            {"damage": damage},
            marker("beat_forge", base=5 + (2 if upgraded else 0), extra=5 + (2 if upgraded else 0)),
        )
    if card_id == "flak_cannon":
        return (marker("flak_cannon", amount=damage),)
    if card_id == "reboot":
        return (marker("reboot"), {"draw": draw})
    if card_id == "scrape":
        return ({"damage": damage}, marker("scrape", count=draw))
    if card_id == "uproar":
        return tuple(step for step in steps if "damage" in step) + (marker("random_attack"),)
    if card_id in {"claw", "maul"}:
        return tuple(step for step in steps if "damage" in step) + (
            marker("grow_matching", amount=number(r"cards by (\d+)", 1)),
        )
    if card_id in {"adaptive_strike", "undeath"}:
        return steps + (marker("clone_discard", zero_cost=card_id == "adaptive_strike"),)
    if card_id == "drain_power":
        return (
            {"damage": damage},
            marker("upgrade_random", zone="discard_pile", count=number(r"upgrade (\d+)", 2)),
        )
    if card_id in {"genetic_algorithm", "the_scythe"}:
        return steps + (
            marker(
                "permanent_growth",
                field="block" if card_id == "genetic_algorithm" else "damage",
                amount=number(r"by (\d+)", 3),
            ),
        )
    if card_id == "brightest_flame":
        return steps + ({"max_hp": -number(r"lose (\d+) max hp", 1)},)
    if card_id == "modded":
        return ({"orb_slot_delta": {"amount": 1}}, {"draw": draw}, {"self_cost_delta": 1})
    if card_id == "momentum_strike":
        return steps + (marker("set_self_cost", cost=0),)
    if card_id in {"make_it_so", "banshees_cry"}:
        return tuple(step for step in steps if any(key in step for key in ("damage", "all_damage")))
    if card_id == "borrowed_time":
        return steps + (marker("install_cost", power="borrowed_time", amount=1, duration="turn"),)
    if card_id in {"synthesis", "veilpiercer"}:
        return steps + (
            marker(
                "install_cost",
                power="free_power" if card_id == "synthesis" else "free_ethereal",
                amount=1,
            ),
        )
    if card_id == "orbit":
        return (marker("install_cost", power="orbit", amount=1),)
    if card_id == "summon_forth":
        return (
            marker("retrieve", zone="any", card_id="sovereign_blade"),
            resource("forge", number(r"forge (\d+)", 8)),
        )
    if card_id == "big_bang":
        return (
            {"draw": draw},
            resource("star", number(r"gain \[star:(\d+)\]")),
            {"energy": number(r"gain \[energy:(\d+)\]")},
            resource("forge", number(r"forge (\d+)")),
        )
    if card_id in {"glow", "scourge", "spoils_of_battle", "pull_aggro", "bulk_up"}:
        order = {
            "glow": ("player_resource", "draw", "next_turn"),
            "scourge": ("apply_status", "draw"),
            "spoils_of_battle": ("player_resource", "draw"),
            "pull_aggro": ("player_resource", "block"),
            "bulk_up": ("orb_slot_delta", "apply_status"),
        }[card_id]
        return tuple(step for key in order for step in steps if key in step)
    if card_id == "rip_and_tear":
        return (
            {
                "repeat_damage": {
                    "amount": damage,
                    "basis": "energy_spent",
                    "extra_hits": 2,
                    "target": "a random enemy",
                    "fixed_hits": 2,
                }
            },
        )
    return steps


def source_class_card_updates(source: Mapping[str, Any]) -> dict[str, Any]:
    """Correct catalog target aliases and source-only play restrictions."""
    card_id = str(source.get("id", source.get("card_id", ""))).lower()
    if card_id in {"energy_surge", "glimpse_beyond", "legion_of_bone"}:
        return {"target": "self"}
    if card_id in {"flak_cannon", "rip_and_tear"}:
        return {"target": "none"}
    return {}

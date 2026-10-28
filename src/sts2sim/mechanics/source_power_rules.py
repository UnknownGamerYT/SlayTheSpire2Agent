"""Stable source power metadata and type classification.

The 260 declarations are an independently recorded crosswalk of public source
properties, not copied game implementations. PowerModel.GetTypeForAmount and
Rend.ShouldCountPower supply the small rules below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .powers import normalize_power_id

PowerKind = Literal["buff", "debuff"]
StackKind = Literal["counter", "single", "none"]


@dataclass(frozen=True, slots=True)
class SourcePowerDefinition:
    power_id: str
    kind: PowerKind
    stack_type: StackKind
    allow_negative: bool = False
    temporary: bool = False


# stable 0.107.1/build23811903, sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7
# MegaCrit.Sts2.Core.Models.Powers/<PascalCaseId>Power.cs.
_DECLARATIONS: tuple[SourcePowerDefinition, ...] = (
    SourcePowerDefinition("accelerant", "buff", "counter"),
    SourcePowerDefinition("accuracy", "buff", "counter"),
    SourcePowerDefinition("adaptable", "buff", "single"),
    SourcePowerDefinition("afterimage", "buff", "counter"),
    SourcePowerDefinition("aggression", "buff", "counter"),
    SourcePowerDefinition("anticipate", "buff", "counter", False, True),
    SourcePowerDefinition("arsenal", "buff", "counter"),
    SourcePowerDefinition("artifact", "buff", "counter"),
    SourcePowerDefinition("asleep", "buff", "counter"),
    SourcePowerDefinition("automation", "buff", "counter"),
    SourcePowerDefinition("back_attack_left", "buff", "single"),
    SourcePowerDefinition("back_attack_right", "buff", "single"),
    SourcePowerDefinition("barricade", "buff", "single"),
    SourcePowerDefinition("battleworn_dummy_time_limit", "buff", "counter"),
    SourcePowerDefinition("beacon_of_hope", "buff", "single"),
    SourcePowerDefinition("biased_cognition", "debuff", "counter"),
    SourcePowerDefinition("black_hole", "buff", "counter"),
    SourcePowerDefinition("block_next_turn", "buff", "counter"),
    SourcePowerDefinition("blur", "buff", "counter"),
    SourcePowerDefinition("borrowed_time", "debuff", "counter"),
    SourcePowerDefinition("buffer", "buff", "counter"),
    SourcePowerDefinition("burrowed", "buff", "single"),
    SourcePowerDefinition("burst", "buff", "counter"),
    SourcePowerDefinition("calamity", "buff", "counter"),
    SourcePowerDefinition("calcify", "buff", "counter"),
    SourcePowerDefinition("call_of_the_void", "buff", "counter"),
    SourcePowerDefinition("chains_of_binding", "debuff", "counter"),
    SourcePowerDefinition("child_of_the_stars", "buff", "counter"),
    SourcePowerDefinition("clarity", "buff", "counter"),
    SourcePowerDefinition("colossus", "buff", "counter"),
    SourcePowerDefinition("confused", "debuff", "single"),
    SourcePowerDefinition("conqueror", "debuff", "counter"),
    SourcePowerDefinition("constrict", "debuff", "counter"),
    SourcePowerDefinition("consuming_shadow", "buff", "counter"),
    SourcePowerDefinition("coolant", "buff", "counter"),
    SourcePowerDefinition("coordinate", "buff", "counter", False, True),
    SourcePowerDefinition("corrosive_wave", "buff", "counter"),
    SourcePowerDefinition("corruption", "buff", "single"),
    SourcePowerDefinition("countdown", "buff", "counter"),
    SourcePowerDefinition("covered", "buff", "single"),
    SourcePowerDefinition("crab_rage", "buff", "single"),
    SourcePowerDefinition("creative_ai", "buff", "counter"),
    SourcePowerDefinition("crimson_mantle", "buff", "counter"),
    SourcePowerDefinition("cruelty", "buff", "counter"),
    SourcePowerDefinition("crush_under", "debuff", "counter", False, True),
    SourcePowerDefinition("curious", "buff", "counter"),
    SourcePowerDefinition("curl_up", "buff", "counter"),
    SourcePowerDefinition("dampen", "debuff", "none"),
    SourcePowerDefinition("danse_macabre", "buff", "counter"),
    SourcePowerDefinition("dark_embrace", "buff", "counter"),
    SourcePowerDefinition("dark_shackles", "debuff", "counter", False, True),
    SourcePowerDefinition("debilitate", "debuff", "counter"),
    SourcePowerDefinition("demesne", "buff", "counter"),
    SourcePowerDefinition("demise", "debuff", "counter"),
    SourcePowerDefinition("demon_form", "buff", "counter"),
    SourcePowerDefinition("devour_life", "buff", "counter"),
    SourcePowerDefinition("dexterity", "buff", "counter", True),
    SourcePowerDefinition("diamond_diadem", "buff", "single"),
    SourcePowerDefinition("die_for_you", "buff", "single"),
    SourcePowerDefinition("disintegration", "debuff", "counter"),
    SourcePowerDefinition("doom", "debuff", "counter"),
    SourcePowerDefinition("double_damage", "buff", "counter"),
    SourcePowerDefinition("draw_cards_next_turn", "buff", "counter"),
    SourcePowerDefinition("duplication", "buff", "counter"),
    SourcePowerDefinition("dying_star", "debuff", "counter", False, True),
    SourcePowerDefinition("echo_form", "buff", "counter"),
    SourcePowerDefinition("energy_next_turn", "buff", "counter"),
    SourcePowerDefinition("enfeebling_touch", "debuff", "counter", False, True),
    SourcePowerDefinition("enrage", "buff", "counter"),
    SourcePowerDefinition("entropy", "buff", "counter"),
    SourcePowerDefinition("envenom", "buff", "counter"),
    SourcePowerDefinition("escape_artist", "buff", "counter"),
    SourcePowerDefinition("fan_of_knives", "buff", "single"),
    SourcePowerDefinition("fasten", "buff", "counter"),
    SourcePowerDefinition("feeding_frenzy", "buff", "counter", False, True),
    SourcePowerDefinition("feel_no_pain", "buff", "counter"),
    SourcePowerDefinition("feral", "buff", "counter"),
    SourcePowerDefinition("flame_barrier", "buff", "counter"),
    SourcePowerDefinition("flanking", "debuff", "counter"),
    SourcePowerDefinition("flex_potion", "buff", "counter", False, True),
    SourcePowerDefinition("flutter", "buff", "counter"),
    SourcePowerDefinition("focus", "buff", "counter", True),
    SourcePowerDefinition("focused_strike", "buff", "counter", False, True),
    SourcePowerDefinition("forbidden_grimoire", "buff", "counter"),
    SourcePowerDefinition("foregone_conclusion", "buff", "counter"),
    SourcePowerDefinition("frail", "debuff", "counter"),
    SourcePowerDefinition("free_attack", "buff", "counter"),
    SourcePowerDefinition("free_power", "buff", "counter"),
    SourcePowerDefinition("free_skill", "buff", "counter"),
    SourcePowerDefinition("friendship", "buff", "counter"),
    SourcePowerDefinition("furnace", "buff", "counter"),
    SourcePowerDefinition("galvanic", "buff", "counter"),
    SourcePowerDefinition("genesis", "buff", "counter"),
    SourcePowerDefinition("gigantification", "buff", "counter"),
    SourcePowerDefinition("gravity", "buff", "counter"),
    SourcePowerDefinition("guarded", "buff", "single"),
    SourcePowerDefinition("hailstorm", "buff", "counter"),
    SourcePowerDefinition("hammer_time", "buff", "single"),
    SourcePowerDefinition("hang", "debuff", "counter"),
    SourcePowerDefinition("hard_to_kill", "buff", "counter"),
    SourcePowerDefinition("hardened_shell", "buff", "counter"),
    SourcePowerDefinition("hatch", "buff", "counter"),
    SourcePowerDefinition("haunt", "buff", "counter"),
    SourcePowerDefinition("heist", "buff", "counter"),
    SourcePowerDefinition("helical_dart", "buff", "counter", False, True),
    SourcePowerDefinition("hello_world", "buff", "counter"),
    SourcePowerDefinition("hellraiser", "buff", "single"),
    SourcePowerDefinition("hex", "debuff", "single"),
    SourcePowerDefinition("high_voltage", "buff", "counter"),
    SourcePowerDefinition("hotfix", "buff", "counter", False, True),
    SourcePowerDefinition("illusion", "buff", "single"),
    SourcePowerDefinition("imbalanced", "debuff", "single"),
    SourcePowerDefinition("improvement", "buff", "counter"),
    SourcePowerDefinition("inferno", "buff", "counter"),
    SourcePowerDefinition("infested", "buff", "single"),
    SourcePowerDefinition("infinite_blades", "buff", "counter"),
    SourcePowerDefinition("intangible", "buff", "counter"),
    SourcePowerDefinition("intercept", "buff", "single"),
    SourcePowerDefinition("iteration", "buff", "counter"),
    SourcePowerDefinition("juggernaut", "buff", "counter"),
    SourcePowerDefinition("juggling", "buff", "counter"),
    SourcePowerDefinition("knockdown", "debuff", "counter"),
    SourcePowerDefinition("leadership", "buff", "counter"),
    SourcePowerDefinition("lethality", "buff", "counter"),
    SourcePowerDefinition("lightning_rod", "buff", "counter"),
    SourcePowerDefinition("loop", "buff", "counter"),
    SourcePowerDefinition("machine_learning", "buff", "counter"),
    SourcePowerDefinition("magic_bomb", "debuff", "counter"),
    SourcePowerDefinition("mangle", "debuff", "counter", False, True),
    SourcePowerDefinition("master_planner", "buff", "single"),
    SourcePowerDefinition("mayhem", "buff", "counter"),
    SourcePowerDefinition("mind_rot", "debuff", "counter"),
    SourcePowerDefinition("minion", "buff", "single"),
    SourcePowerDefinition("monarchs_gaze", "buff", "counter"),
    SourcePowerDefinition("monarchs_gaze_strength_down", "debuff", "counter", False, True),
    SourcePowerDefinition("monologue", "buff", "none"),
    SourcePowerDefinition("necro_mastery", "buff", "counter"),
    SourcePowerDefinition("nemesis", "buff", "single"),
    SourcePowerDefinition("neurosurge", "debuff", "counter"),
    SourcePowerDefinition("nightmare", "buff", "counter"),
    SourcePowerDefinition("no_block", "debuff", "counter"),
    SourcePowerDefinition("no_draw", "debuff", "single"),
    SourcePowerDefinition("no_energy_gain", "debuff", "single"),
    SourcePowerDefinition("nostalgia", "buff", "counter"),
    SourcePowerDefinition("noxious_fumes", "buff", "counter"),
    SourcePowerDefinition("oblivion", "debuff", "counter"),
    SourcePowerDefinition("one_two_punch", "buff", "counter"),
    SourcePowerDefinition("orbit", "buff", "counter"),
    SourcePowerDefinition("outbreak", "buff", "counter"),
    SourcePowerDefinition("pagestorm", "buff", "counter"),
    SourcePowerDefinition("painful_stabs", "buff", "counter"),
    SourcePowerDefinition("pale_blue_dot", "buff", "counter"),
    SourcePowerDefinition("panache", "buff", "counter"),
    SourcePowerDefinition("paper_cuts", "buff", "counter"),
    SourcePowerDefinition("parry", "buff", "counter"),
    SourcePowerDefinition("personal_hive", "buff", "counter"),
    SourcePowerDefinition("phantom_blades", "buff", "counter"),
    SourcePowerDefinition("piercing_wail", "debuff", "counter", False, True),
    SourcePowerDefinition("pillar_of_creation", "buff", "counter"),
    SourcePowerDefinition("plating", "buff", "counter"),
    SourcePowerDefinition("plow", "debuff", "counter"),
    SourcePowerDefinition("poison", "debuff", "counter"),
    SourcePowerDefinition("possess_speed", "buff", "single"),
    SourcePowerDefinition("possess_strength", "buff", "single"),
    SourcePowerDefinition("prep_time", "buff", "counter"),
    SourcePowerDefinition("pyre", "buff", "counter"),
    SourcePowerDefinition("radiance", "buff", "counter"),
    SourcePowerDefinition("rage", "buff", "counter"),
    SourcePowerDefinition("rampart", "buff", "counter"),
    SourcePowerDefinition("ravenous", "buff", "counter"),
    SourcePowerDefinition("reaper_form", "buff", "counter"),
    SourcePowerDefinition("reattach", "buff", "single"),
    SourcePowerDefinition("rebound", "buff", "counter"),
    SourcePowerDefinition("reflect", "buff", "counter"),
    SourcePowerDefinition("regen", "buff", "counter"),
    SourcePowerDefinition("reptile_trinket", "buff", "counter", False, True),
    SourcePowerDefinition("retain_hand", "buff", "counter"),
    SourcePowerDefinition("ringing", "debuff", "single"),
    SourcePowerDefinition("ritual", "buff", "counter"),
    SourcePowerDefinition("rolling_boulder", "buff", "counter"),
    SourcePowerDefinition("royalties", "buff", "counter"),
    SourcePowerDefinition("rupture", "buff", "counter"),
    SourcePowerDefinition("sandpit", "buff", "counter"),
    SourcePowerDefinition("seeking_edge", "buff", "single"),
    SourcePowerDefinition("self_forming_clay", "buff", "counter"),
    SourcePowerDefinition("sentry_mode", "buff", "counter"),
    SourcePowerDefinition("serpent_form", "buff", "counter"),
    SourcePowerDefinition("setup_strike", "buff", "counter", False, True),
    SourcePowerDefinition("shackling_potion", "debuff", "counter", False, True),
    SourcePowerDefinition("shadow_step", "buff", "counter"),
    SourcePowerDefinition("shadowmeld", "buff", "counter"),
    SourcePowerDefinition("shriek", "debuff", "counter", True),
    SourcePowerDefinition("shrink", "debuff", "counter", True),
    SourcePowerDefinition("shroud", "buff", "counter"),
    SourcePowerDefinition("sic_em", "debuff", "counter"),
    SourcePowerDefinition("signal_boost", "buff", "counter"),
    SourcePowerDefinition("skittish", "buff", "counter"),
    SourcePowerDefinition("sleight_of_flesh", "buff", "counter"),
    SourcePowerDefinition("slippery", "buff", "counter"),
    SourcePowerDefinition("sloth", "debuff", "counter"),
    SourcePowerDefinition("slow", "debuff", "counter"),
    SourcePowerDefinition("slumber", "buff", "counter"),
    SourcePowerDefinition("smoggy", "debuff", "single"),
    SourcePowerDefinition("smokestack", "buff", "counter"),
    SourcePowerDefinition("sneaky", "buff", "counter"),
    SourcePowerDefinition("soar", "buff", "single"),
    SourcePowerDefinition("spectrum_shift", "buff", "counter"),
    SourcePowerDefinition("speed_potion", "buff", "counter", False, True),
    SourcePowerDefinition("speedster", "buff", "counter"),
    SourcePowerDefinition("spinner", "buff", "counter"),
    SourcePowerDefinition("spirit_of_ash", "buff", "counter"),
    SourcePowerDefinition("stampede", "buff", "counter"),
    SourcePowerDefinition("star_next_turn", "buff", "counter"),
    SourcePowerDefinition("steam_eruption", "buff", "counter"),
    SourcePowerDefinition("stock", "buff", "counter"),
    SourcePowerDefinition("storm", "buff", "counter"),
    SourcePowerDefinition("strangle", "debuff", "counter"),
    SourcePowerDefinition("stratagem", "buff", "counter"),
    SourcePowerDefinition("strength", "buff", "counter", True),
    SourcePowerDefinition("subroutine", "buff", "counter"),
    SourcePowerDefinition("suck", "buff", "counter"),
    SourcePowerDefinition("summon_next_turn", "buff", "counter"),
    SourcePowerDefinition("surprise", "buff", "single"),
    SourcePowerDefinition("surrounded", "debuff", "single"),
    SourcePowerDefinition("swipe", "buff", "single"),
    SourcePowerDefinition("sword_sage", "buff", "counter"),
    SourcePowerDefinition("synchronize", "buff", "counter", False, True),
    SourcePowerDefinition("tag_team", "debuff", "counter"),
    SourcePowerDefinition("tainted", "debuff", "counter"),
    SourcePowerDefinition("tangled", "debuff", "counter"),
    SourcePowerDefinition("tank", "buff", "single"),
    SourcePowerDefinition("temporary_dexterity", "buff", "counter", False, True),
    SourcePowerDefinition("temporary_focus", "buff", "counter", False, True),
    SourcePowerDefinition("temporary_strength", "buff", "counter", False, True),
    SourcePowerDefinition("tender", "debuff", "counter"),
    SourcePowerDefinition("territorial", "buff", "counter"),
    SourcePowerDefinition("the_bomb", "buff", "counter"),
    SourcePowerDefinition("the_gambit", "debuff", "single"),
    SourcePowerDefinition("the_hunt", "buff", "counter"),
    SourcePowerDefinition("the_sealed_throne", "buff", "counter"),
    SourcePowerDefinition("thievery", "buff", "counter"),
    SourcePowerDefinition("thorns", "buff", "counter"),
    SourcePowerDefinition("thunder", "buff", "counter"),
    SourcePowerDefinition("tools_of_the_trade", "buff", "counter"),
    SourcePowerDefinition("toric_toughness", "buff", "counter"),
    SourcePowerDefinition("tracking", "buff", "counter"),
    SourcePowerDefinition("trash_to_treasure", "buff", "counter"),
    SourcePowerDefinition("tyranny", "buff", "counter"),
    SourcePowerDefinition("unmovable", "buff", "counter"),
    SourcePowerDefinition("veilpiercer", "buff", "counter"),
    SourcePowerDefinition("vicious", "buff", "counter"),
    SourcePowerDefinition("vigor", "buff", "counter"),
    SourcePowerDefinition("vital_spark", "buff", "counter"),
    SourcePowerDefinition("void_form", "buff", "counter"),
    SourcePowerDefinition("vulnerable", "debuff", "counter"),
    SourcePowerDefinition("waste_away", "debuff", "counter"),
    SourcePowerDefinition("weak", "debuff", "counter"),
    SourcePowerDefinition("well_laid_plans", "buff", "counter"),
    SourcePowerDefinition("withering_presence", "buff", "counter"),
    SourcePowerDefinition("wraith_form", "debuff", "counter"),
)


def _key(value: object) -> str:
    return normalize_power_id(value).replace("_", "").replace(" ", "")


SOURCE_POWER_DEFINITIONS: Mapping[str, SourcePowerDefinition] = MappingProxyType(
    {definition.power_id: definition for definition in _DECLARATIONS}
)
_BY_COMPACT_ID = {_key(definition.power_id): definition for definition in _DECLARATIONS}

# The simulator uses these names for the corresponding concrete source powers.
_ALIASES = {
    "nocardblockgain": "noblock",
    "nodrawthisturn": "nodraw",
    "noenergygainthisturn": "noenergygain",
    "confusion": "confused",
    "strengthdown": "temporarystrength",
    "dexteritydown": "temporarydexterity",
}
# These inline modifiers represent an underlying signed stat together with its
# temporary wrapper. Their stored negative amount denotes a temporary stat loss.
_SIGNED_TEMPORARY_IDS = frozenset({"temporarystrength", "temporarydexterity", "temporaryfocus"})
_RECOVERY_IDS = frozenset({"strengthdown", "dexteritydown"})


def source_power_definition(power_id: object) -> SourcePowerDefinition | None:
    key = _key(power_id)
    return _BY_COMPACT_ID.get(_ALIASES.get(key, key))


def power_type_for_amount(power_id: object, amount: int) -> PowerKind:
    """Classify the application delta, as Artifact does, or a current amount."""
    key = _key(power_id)
    if key in _SIGNED_TEMPORARY_IDS:
        return "debuff" if amount < 0 else "buff"
    definition = source_power_definition(key)
    if definition is None:
        return "buff"
    if definition.stack_type == "counter" and definition.allow_negative and amount < 0:
        return "debuff"
    if not definition.allow_negative and definition.kind == "debuff" and amount < 0:
        return "buff"
    return definition.kind


def power_is_temporary(power_id: object) -> bool:
    definition = source_power_definition(power_id)
    return definition.temporary if definition is not None else False


def power_allows_negative(power_id: object) -> bool:
    key = _key(power_id)
    if key in _SIGNED_TEMPORARY_IDS:
        return True
    definition = source_power_definition(key)
    return definition.allow_negative if definition is not None else True


def power_application_is_debuff(power_id: object, amount: int) -> bool:
    return amount != 0 and power_type_for_amount(power_id, amount) == "debuff"


def power_triggers_debuff_damage(power_id: object, amount: int) -> bool:
    # Inline temporary-stat modifiers also represent the actual signed stat
    # application. Concrete source wrapper powers are excluded by ITemporaryPower.
    return power_application_is_debuff(power_id, amount) and (
        not power_is_temporary(power_id) or _key(power_id) in _SIGNED_TEMPORARY_IDS
    )


def unique_source_debuff_count(statuses: Mapping[str, int]) -> int:
    amounts: dict[str, int] = {}
    for raw_id, amount in statuses.items():
        key = _key(raw_id)
        if key in _RECOVERY_IDS:
            continue
        definition = source_power_definition(key)
        canonical = _key(definition.power_id) if definition is not None else key
        amounts[canonical] = amounts.get(canonical, 0) + amount
    # Inline temporary stats are kept separately by this simulator. In source,
    # the signed Strength/Dexterity/Focus power already includes that modifier.
    for stat in ("strength", "dexterity", "focus"):
        amounts[stat] = amounts.get(stat, 0) + amounts.get("temporary" + stat, 0)
    return sum(
        amount != 0
        and not power_is_temporary(power_id)
        and source_power_definition(power_id) is not None
        and power_type_for_amount(power_id, amount) == "debuff"
        for power_id, amount in amounts.items()
    )

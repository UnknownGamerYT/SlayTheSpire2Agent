"""Source-backed catalog of simple event option primitives.

The engine has its own transition parser for event rooms.  This module stays
below that layer: it turns cached Codex event option text into
``EventOption`` primitives where the current mechanics model can represent the
effect, and it records explicit coverage metadata for the parts it cannot.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sts2sim.content.event_effects import (
    EventTextAmount,
    EventTextEffect,
    EventTextEffectKind,
    normalize_event_option_description,
    parse_event_option_effects,
)
from sts2sim.mechanics.event_rooms import EventOption

EventCatalogStatus = Literal["supported", "partial", "unsupported"]


@dataclass(frozen=True, slots=True)
class EventCatalogOptionCoverage:
    """Coverage marker for one cached event option."""

    event_id: str
    option_id: str
    event_name: str
    option_title: str
    act: str | None
    status: EventCatalogStatus
    categories: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    description: str = ""
    effect_kinds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ContentLookups:
    cards: Mapping[str, str]
    relics: Mapping[str, str]
    potions: Mapping[str, str]


_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "eng"

_EVENT_ALIASES = {
    "lantern_key": "the_lantern_key",
    "war_historian": "war_historian_repy",
    "repy": "war_historian_repy",
}

_CATEGORY_MESSAGES = {
    "all_gold": "Gold amount is ALL and cannot be represented as a fixed delta.",
    "card_reward": "Card reward generation requires a card reward picker.",
    "character_card_reward": "Character-specific card reward generation is not a fixed card add.",
    "combat_reward_hint": "Combat rewards are hinted by text but not modeled as immediate rewards.",
    "custom_card": "Custom card creation is bespoke.",
    "debt": "Debt-like side effect requires bespoke event handling.",
    "deck_duplicate": "Deck duplication is not an EventOption primitive.",
    "delayed_combat_effect": "Effect applies after future combat timing.",
    "divine": "Divine action is bespoke and not represented by EventOption.",
    "downgrade": "Card downgrade is not an EventOption primitive.",
    "draw_pile_shuffle": "Draw-pile insertion is outside current event primitives.",
    "empty_option": "Option has no primitive description.",
    "enchant": "Card enchantment is a bespoke mechanic.",
    "fixed_card_unknown": "Fixed card name could not be resolved to a cached card id.",
    "fixed_potion_unknown": "Fixed potion name could not be resolved to a cached potion id.",
    "fixed_relic_unknown": "Fixed relic name could not be resolved to a cached relic id.",
    "gold_gain_range": "Gold gain is a range and cannot be represented deterministically.",
    "gold_loss_range": "Gold loss is a range and cannot be represented deterministically.",
    "locked_option": "Locked or requirement-only option is not a selectable primitive.",
    "max_hp_set": "Setting max HP directly is not an EventOption primitive.",
    "potion_loss": "Potion loss is not an EventOption primitive.",
    "qualified_random_potion_pool": "Potion rarity qualifier is metadata only.",
    "qualified_random_relic_pool": "Relic pool qualifier is metadata only.",
    "random_card_add": "Random card add requires card reward generation.",
    "random_power_card": "Random Power card generation is not a fixed card add.",
    "relic_choice": "Relic choice UI is not represented by a single random relic draw.",
    "relic_trade": "Relic trading requires removing a selected relic.",
    "stateful_followup_options": "Choosing this option opens additional event options.",
    "transform_filter_metadata_only": "Transform filter is recorded as metadata only.",
    "transform_target_metadata_only": "Transform target is recorded as metadata only.",
    "trial_role": "Trial role behavior is bespoke.",
    "unparsed_text": "No known primitive effect could be parsed from the option text.",
}

_OPTION_OVERRIDES: Mapping[tuple[str, str], Mapping[str, object]] = {
    (
        "the_lantern_key",
        "keep_the_key",
    ): {
        "fixed_card_ids": ("lantern_key",),
        "combat_encounter": "normal",
        "metadata": {"reward_timing": "post_combat"},
    },
}


def known_event_catalog_options(event_id: str) -> tuple[EventOption, ...]:
    """Return cataloged primitive options for a known cached event id.

    Known events with no representable primitive options return an empty tuple.
    Unknown event ids raise ``ValueError`` so typos do not look like unsupported
    coverage.
    """

    key = _event_key(event_id)
    if key not in _ALL_EVENT_IDS:
        raise ValueError(f"Unknown event id: {event_id}")
    return _CATALOG.get(key, ())


def event_catalog_ids() -> tuple[str, ...]:
    """Return normalized event ids with at least one catalog option."""

    return tuple(sorted(_CATALOG))


def event_catalog_coverage(event_id: str | None = None) -> tuple[EventCatalogOptionCoverage, ...]:
    """Return source-backed coverage rows for cached event options."""

    if event_id is None:
        return _COVERAGE
    key = _event_key(event_id)
    if key not in _ALL_EVENT_IDS:
        raise ValueError(f"Unknown event id: {event_id}")
    return tuple(row for row in _COVERAGE if row.event_id == key)


def event_catalog_unsupported_options(
    event_id: str | None = None,
) -> tuple[EventCatalogOptionCoverage, ...]:
    """Return coverage rows that are partial or unsupported."""

    return tuple(row for row in event_catalog_coverage(event_id) if row.status != "supported")


def event_catalog_unsupported_categories() -> tuple[str, ...]:
    """Return the distinct unsupported/bespoke category names in coverage."""

    return tuple(
        sorted(
            {
                category
                for row in event_catalog_unsupported_options()
                for category in row.categories
            }
        )
    )


def _build_event_catalog() -> tuple[
    dict[str, tuple[EventOption, ...]],
    tuple[EventCatalogOptionCoverage, ...],
    frozenset[str],
]:
    lookups = _build_content_lookups()
    catalog: dict[str, tuple[EventOption, ...]] = {}
    coverage: list[EventCatalogOptionCoverage] = []
    all_event_ids: set[str] = set()

    for event in _load_rows("events"):
        source_event_id = str(event.get("id", ""))
        event_id = _normalized_id(source_event_id)
        if not event_id:
            continue
        all_event_ids.add(event_id)

        options: list[EventOption] = []
        for option in _initial_options(event):
            converted, marker = _convert_option(
                event,
                option,
                lookups,
                has_followup_options=_has_followup_options(event, option),
            )
            coverage.append(marker)
            if converted is not None:
                options.append(converted)

        if options:
            catalog[event_id] = tuple(options)

    return catalog, tuple(coverage), frozenset(all_event_ids)


def _build_content_lookups() -> _ContentLookups:
    return _ContentLookups(
        cards=_content_lookup("cards"),
        relics=_content_lookup("relics"),
        potions=_content_lookup("potions"),
    )


def _content_lookup(dataset: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in _load_rows(dataset):
        raw_id = row.get("id", row.get(f"{dataset[:-1]}_id"))
        if raw_id is None:
            continue
        content_id = _normalized_id(raw_id)
        for value in _name_values(row):
            for key in _name_keys(value):
                lookup.setdefault(key, content_id)
        lookup.setdefault(_name_key(raw_id), content_id)
    return lookup


def _name_values(row: Mapping[str, Any]) -> tuple[object, ...]:
    values: list[object] = []
    for key in ("name", "id"):
        value = row.get(key)
        if value is not None:
            values.append(value)

    for key in ("name_variants", "names"):
        value = row.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values.extend(value)

    return tuple(values)


def _convert_option(
    event: Mapping[str, Any],
    option: Mapping[str, Any],
    lookups: _ContentLookups,
    *,
    has_followup_options: bool,
) -> tuple[EventOption | None, EventCatalogOptionCoverage]:
    event_id = _normalized_id(str(event.get("id", "")))
    event_name = str(event.get("name", event_id))
    act = _optional_str(event.get("act"))
    source_option_id = str(option.get("id", option.get("title", "option")))
    option_id = _normalized_id(source_option_id)
    title = str(option.get("title", source_option_id))
    description = normalize_event_option_description(str(option.get("description", "")))
    effects = parse_event_option_effects(description)
    effect_kinds = tuple(effect.kind.value for effect in effects)

    categories = set(_text_categories(option_id, title, description))
    if has_followup_options:
        categories.add("stateful_followup_options")

    metadata: dict[str, object] = {
        "source_event_id": str(event.get("id", "")),
        "source_option_id": source_option_id,
        "source_event_name": event_name,
        "source_act": act,
        "source_description": description,
        "effect_kinds": effect_kinds,
    }

    gold_delta = 0
    hp_delta = 0
    heal_percent_max_hp = 0.0
    max_hp_delta = 0
    fixed_relic_ids: list[str] = []
    random_relic_count = 0
    fixed_potion_ids: list[str] = []
    random_potion_count = 0
    fixed_card_ids: list[str] = []
    remove_card_ids: list[str] = []
    required_card_ids: list[str] = []
    required_relic_ids: list[str] = []
    upgrade_random_count = 0
    transform_random_count = 0
    remove_random_count = 0
    combat_encounter: str | None = None
    primitive_count = 0

    for effect in effects:
        if effect.kind is EventTextEffectKind.GOLD_GAIN:
            value = _amount_int(effect.amount)
            if value is None:
                _record_amount_gap("gold_gain", effect.amount, metadata, categories)
            else:
                gold_delta += value
                primitive_count += 1
        elif effect.kind is EventTextEffectKind.GOLD_LOSS:
            value = _amount_int(effect.amount)
            if value is None:
                _record_amount_gap("gold_loss", effect.amount, metadata, categories)
            else:
                gold_delta -= value
                metadata["gold_cost"] = abs(gold_delta) if gold_delta < 0 else value
                primitive_count += 1
        elif effect.kind is EventTextEffectKind.HP_LOSS:
            value = _amount_int(effect.amount)
            if value is None:
                _record_amount_gap("hp_loss", effect.amount, metadata, categories)
            else:
                hp_delta -= value
                primitive_count += 1
        elif effect.kind is EventTextEffectKind.HP_HEAL:
            heal_added = _apply_heal_effect(effect, metadata, categories)
            hp_delta += heal_added[0]
            heal_percent_max_hp += heal_added[1]
            primitive_count += heal_added[2]
        elif effect.kind is EventTextEffectKind.MAX_HP_GAIN:
            value = _amount_int(effect.amount)
            if value is None:
                _record_amount_gap("max_hp_gain", effect.amount, metadata, categories)
            else:
                max_hp_delta += value
                primitive_count += 1
        elif effect.kind is EventTextEffectKind.MAX_HP_LOSS:
            value = _amount_int(effect.amount)
            if value is None:
                _record_amount_gap("max_hp_loss", effect.amount, metadata, categories)
            else:
                max_hp_delta -= value
                primitive_count += 1
        elif effect.kind is EventTextEffectKind.MAX_HP_SET:
            categories.add("max_hp_set")
        elif effect.kind is EventTextEffectKind.RELIC_OBTAIN:
            relic_added = _apply_relic_effect(
                effect,
                lookups,
                metadata,
                categories,
                fixed_relic_ids,
                fixed_card_ids,
            )
            random_relic_count += relic_added[0]
            primitive_count += relic_added[1]
        elif effect.kind is EventTextEffectKind.POTION_PROCURE:
            potion_added = _apply_potion_effect(
                effect,
                lookups,
                metadata,
                categories,
                fixed_potion_ids,
            )
            random_potion_count += potion_added[0]
            primitive_count += potion_added[1]
        elif effect.kind is EventTextEffectKind.POTION_LOSS:
            categories.add("potion_loss")
        elif effect.kind is EventTextEffectKind.CARD_ADD:
            primitive_count += _apply_card_add_effect(
                effect,
                lookups,
                metadata,
                categories,
                fixed_card_ids,
            )
        elif effect.kind is EventTextEffectKind.CARD_REMOVE:
            remove_added = _apply_card_remove_effect(
                effect,
                lookups,
                metadata,
                categories,
                remove_card_ids,
                required_card_ids,
            )
            remove_random_count += remove_added[0]
            primitive_count += remove_added[1]
        elif effect.kind is EventTextEffectKind.CARD_UPGRADE:
            upgrade_added = _apply_card_upgrade_effect(effect, metadata, categories)
            upgrade_random_count += upgrade_added[0]
            primitive_count += upgrade_added[1]
        elif effect.kind is EventTextEffectKind.CARD_TRANSFORM:
            transform_added = _apply_card_transform_effect(effect, lookups, metadata, categories)
            transform_random_count += transform_added[0]
            primitive_count += transform_added[1]
        elif effect.kind is EventTextEffectKind.FIGHT:
            combat_encounter = _combat_encounter(event_id, effect)
            _record_fight_metadata(effect, metadata, categories)
            primitive_count += 1

    extra_card_ids = _extra_fixed_card_gain_ids(description, lookups)
    if extra_card_ids:
        fixed_card_ids.extend(extra_card_ids)
        primitive_count += len(extra_card_ids)

    override = _OPTION_OVERRIDES.get((event_id, option_id))
    if override is not None:
        primitive_count += _apply_override(
            override,
            fixed_card_ids,
            fixed_relic_ids,
            fixed_potion_ids,
            remove_card_ids,
            required_card_ids,
            required_relic_ids,
            metadata,
        )
        combat_encounter = str(override.get("combat_encounter") or combat_encounter or "")
        if not combat_encounter:
            combat_encounter = None

    if not effects and not extra_card_ids:
        categories.add("empty_option" if not description else "unparsed_text")

    option_obj: EventOption | None = None
    if primitive_count > 0 and "locked_option" not in categories:
        status: EventCatalogStatus = "partial" if categories else "supported"
        metadata["catalog_status"] = status
        metadata["unsupported_categories"] = tuple(sorted(categories))
        option_obj = EventOption(
            option_id=option_id,
            label=title,
            description=description,
            gold_delta=gold_delta,
            hp_delta=hp_delta,
            heal_percent_max_hp=heal_percent_max_hp,
            max_hp_delta=max_hp_delta,
            fixed_relic_ids=tuple(fixed_relic_ids),
            random_relic_count=random_relic_count,
            fixed_potion_ids=tuple(fixed_potion_ids),
            random_potion_count=random_potion_count,
            fixed_card_ids=tuple(fixed_card_ids),
            remove_card_ids=tuple(remove_card_ids),
            required_card_ids=tuple(required_card_ids),
            required_relic_ids=tuple(required_relic_ids),
            upgrade_random_count=upgrade_random_count,
            transform_random_count=transform_random_count,
            remove_random_count=remove_random_count,
            combat_encounter=combat_encounter,
            metadata=metadata,
        )
    else:
        status = "unsupported"

    marker = EventCatalogOptionCoverage(
        event_id=event_id,
        option_id=option_id,
        event_name=event_name,
        option_title=title,
        act=act,
        status=status,
        categories=tuple(sorted(categories)),
        reasons=tuple(
            _CATEGORY_MESSAGES.get(category, category) for category in sorted(categories)
        ),
        description=description,
        effect_kinds=effect_kinds,
    )
    return option_obj, marker


def _apply_heal_effect(
    effect: EventTextEffect,
    metadata: dict[str, object],
    categories: set[str],
) -> tuple[int, float, int]:
    amount = effect.amount
    if amount is None:
        categories.add("hp_heal_amount_missing")
        return 0, 0.0, 0
    if amount.full:
        return 0, 1.0, 1
    if amount.percent is not None:
        return 0, amount.percent / 100.0, 1
    if amount.value is not None:
        return amount.value, 0.0, 1
    _record_amount_gap("hp_heal", amount, metadata, categories)
    return 0, 0.0, 0


def _apply_relic_effect(
    effect: EventTextEffect,
    lookups: _ContentLookups,
    metadata: dict[str, object],
    categories: set[str],
    fixed_relic_ids: list[str],
    fixed_card_ids: list[str],
) -> tuple[int, int]:
    count = _effect_count(effect)
    if effect.random:
        if effect.qualifier:
            _append_metadata(metadata, "random_relic_qualifiers", effect.qualifier)
            categories.add("qualified_random_relic_pool")
        return count, 1

    if not effect.item_name:
        categories.add("fixed_relic_unknown")
        return 0, 0

    relic_id = _lookup_content_id(lookups.relics, effect.item_name)
    if relic_id is not None:
        fixed_relic_ids.extend([relic_id] * count)
        return 0, 1

    card_id = _lookup_content_id(lookups.cards, effect.item_name)
    if card_id is not None:
        fixed_card_ids.extend([card_id] * count)
        _append_metadata(metadata, "relic_text_resolved_as_card_ids", card_id)
        return 0, 1

    _append_metadata(metadata, "unknown_fixed_relic_names", effect.item_name)
    categories.add("fixed_relic_unknown")
    return 0, 0


def _apply_potion_effect(
    effect: EventTextEffect,
    lookups: _ContentLookups,
    metadata: dict[str, object],
    categories: set[str],
    fixed_potion_ids: list[str],
) -> tuple[int, int]:
    count = _effect_count(effect)
    if effect.random:
        if effect.qualifier:
            _append_metadata(metadata, "random_potion_qualifiers", effect.qualifier)
            categories.add("qualified_random_potion_pool")
        return count, 1

    if not effect.item_name:
        categories.add("fixed_potion_unknown")
        return 0, 0

    potion_id = _lookup_content_id(lookups.potions, effect.item_name)
    if potion_id is None:
        _append_metadata(metadata, "unknown_fixed_potion_names", effect.item_name)
        categories.add("fixed_potion_unknown")
        return 0, 0

    fixed_potion_ids.extend([potion_id] * count)
    return 0, 1


def _apply_card_add_effect(
    effect: EventTextEffect,
    lookups: _ContentLookups,
    metadata: dict[str, object],
    categories: set[str],
    fixed_card_ids: list[str],
) -> int:
    count = _effect_count(effect)
    if effect.random:
        _append_metadata(
            metadata,
            "random_card_adds",
            {
                "count": count,
                "qualifier": effect.qualifier,
                "target": effect.target,
            },
        )
        categories.add("random_card_add")
        return 0

    if not effect.item_name:
        categories.add("fixed_card_unknown")
        return 0

    card_id = _lookup_content_id(lookups.cards, effect.item_name)
    if card_id is None:
        _append_metadata(metadata, "unknown_fixed_card_names", effect.item_name)
        categories.add("fixed_card_unknown")
        return 0

    fixed_card_ids.extend([card_id] * count)
    return 1


def _apply_card_remove_effect(
    effect: EventTextEffect,
    lookups: _ContentLookups,
    metadata: dict[str, object],
    categories: set[str],
    remove_card_ids: list[str],
    required_card_ids: list[str],
) -> tuple[int, int]:
    count = _effect_count(effect)
    name = effect.item_name or ""
    if effect.random or not name or _is_generic_card_name(name):
        _append_metadata(
            metadata,
            "remove_card_markers",
            {"count": count, "selection": "random" if effect.random else "chosen"},
        )
        return count, 1

    card_id = _lookup_content_id(lookups.cards, name)
    if card_id is None:
        _append_metadata(metadata, "unknown_remove_card_names", name)
        categories.add("fixed_card_unknown")
        return 0, 0

    remove_card_ids.extend([card_id] * count)
    required_card_ids.extend([card_id] * count)
    if count > 1:
        metadata["required_card_counts"] = {card_id: count}
    return 0, 1


def _apply_card_upgrade_effect(
    effect: EventTextEffect,
    metadata: dict[str, object],
    categories: set[str],
) -> tuple[int, int]:
    if effect.amount is not None and effect.amount.all:
        categories.add("all_card_upgrade")
        return 0, 0

    count = _effect_count(effect)
    _append_metadata(
        metadata,
        "upgrade_card_markers",
        {"count": count, "selection": "random" if effect.random else "chosen"},
    )
    return count, 1


def _apply_card_transform_effect(
    effect: EventTextEffect,
    lookups: _ContentLookups,
    metadata: dict[str, object],
    categories: set[str],
) -> tuple[int, int]:
    count = _effect_count(effect)
    if effect.qualifier:
        metadata["transform_filter"] = effect.qualifier
        categories.add("transform_filter_metadata_only")
    if effect.target and _normalized_id(effect.target) != "deck":
        target_id = _lookup_content_id(lookups.cards, effect.target) or _normalized_id(
            effect.target
        )
        metadata["transform_target_card_id"] = target_id
        categories.add("transform_target_metadata_only")
    _append_metadata(metadata, "transform_card_markers", {"count": count})
    return count, 1


def _apply_override(
    override: Mapping[str, object],
    fixed_card_ids: list[str],
    fixed_relic_ids: list[str],
    fixed_potion_ids: list[str],
    remove_card_ids: list[str],
    required_card_ids: list[str],
    required_relic_ids: list[str],
    metadata: dict[str, object],
) -> int:
    added = 0
    for key, target in (
        ("fixed_card_ids", fixed_card_ids),
        ("fixed_relic_ids", fixed_relic_ids),
        ("fixed_potion_ids", fixed_potion_ids),
        ("remove_card_ids", remove_card_ids),
        ("required_card_ids", required_card_ids),
        ("required_relic_ids", required_relic_ids),
    ):
        values = _str_sequence(override.get(key))
        if values:
            target.extend(_normalized_id(value) for value in values)
            added += 1

    raw_metadata = override.get("metadata")
    if isinstance(raw_metadata, Mapping):
        metadata.update({str(key): value for key, value in raw_metadata.items()})

    return added


def _record_fight_metadata(
    effect: EventTextEffect,
    metadata: dict[str, object],
    categories: set[str],
) -> None:
    if effect.amount is not None and effect.amount.value is not None:
        metadata["combat_target_hp"] = effect.amount.value
    if effect.target:
        metadata["combat_target"] = effect.target
    if effect.qualifier:
        metadata["combat_reward_hint"] = effect.qualifier
        categories.add("combat_reward_hint")


def _combat_encounter(event_id: str, effect: EventTextEffect) -> str:
    if event_id == "battleworn_dummy" or _normalized_id(effect.target or "") == "dummy":
        return "battleworn_dummy"
    return "normal"


def _extra_fixed_card_gain_ids(description: str, lookups: _ContentLookups) -> tuple[str, ...]:
    card_ids: list[str] = []
    for match in re.finditer(
        r"\bGain\s+(?:an?\s+|the\s+)?(?P<name>[A-Z][A-Za-z' -]+)\b",
        description,
    ):
        name = match.group("name").strip()
        card_id = _lookup_content_id(lookups.cards, name)
        if card_id is not None:
            card_ids.append(card_id)
    return tuple(dict.fromkeys(card_ids))


def _text_categories(option_id: str, title: str, description: str) -> tuple[str, ...]:
    lower = description.lower()
    categories: set[str] = set()
    title_key = _normalized_id(title)

    if (
        option_id.endswith("_locked")
        or title_key == "locked"
        or lower.startswith("requires ")
        or lower.startswith("not enough ")
        or "don't have any" in lower
        or lower.startswith("none of your")
        or lower.startswith("you have no ")
    ):
        categories.add("locked_option")
    if "enchant" in lower:
        categories.add("enchant")
    if "divine" in lower:
        categories.add("divine")
    if "debt" in lower:
        categories.add("debt")
    if "custom card" in lower:
        categories.add("custom_card")
    if "colorless card reward" in lower or " card reward" in lower:
        categories.add("card_reward")
    if "random power" in lower or "random 0 cost card" in lower:
        categories.add("random_power_card")
    if "downgrade" in lower:
        categories.add("downgrade")
    if "duplicate your deck" in lower:
        categories.add("deck_duplicate")
    if "after " in lower and "combat" in lower:
        categories.add("delayed_combat_effect")
    if "start of the next combat" in lower or "start of the next " in lower:
        categories.add("delayed_combat_effect")
    if "shuffle" in lower and "draw pile" in lower:
        categories.add("draw_pile_shuffle")
    if "serve as today" in lower or "not allowed to reject" in lower:
        categories.add("trial_role")
    if "trade " in lower and "relic" in lower:
        categories.add("relic_trade")
    if "choose 1 of" in lower and "doll relic" in lower:
        categories.add("relic_choice")
    if " random cards to add" in lower:
        categories.add("random_card_add")
    if re.search(r"\bobtain\s+\d+\s+[a-z]+\s+cards\b", lower):
        categories.add("character_card_reward")

    return tuple(sorted(categories))


def _record_amount_gap(
    prefix: str,
    amount: EventTextAmount | None,
    metadata: dict[str, object],
    categories: set[str],
) -> None:
    if amount is None:
        categories.add(f"{prefix}_amount_missing")
        return
    if amount.minimum is not None and amount.maximum is not None:
        metadata[f"{prefix}_range"] = (amount.minimum, amount.maximum)
        categories.add(f"{prefix}_range")
    elif amount.all:
        metadata[f"{prefix}_all"] = True
        categories.add("all_gold" if "gold" in prefix else f"{prefix}_all")
    elif amount.or_more:
        metadata[f"{prefix}_minimum"] = amount.minimum
        categories.add(f"{prefix}_or_more")
    elif amount.percent is not None:
        metadata[f"{prefix}_percent"] = amount.percent
        categories.add(f"{prefix}_percent")
    else:
        categories.add(f"{prefix}_amount_missing")


def _amount_int(amount: EventTextAmount | None) -> int | None:
    if amount is None:
        return None
    return amount.value


def _effect_count(effect: EventTextEffect) -> int:
    return max(1, int(effect.count or 1))


def _append_metadata(metadata: dict[str, object], key: str, value: object) -> None:
    previous = metadata.get(key, ())
    if isinstance(previous, tuple):
        metadata[key] = previous + (value,)
    else:
        metadata[key] = (previous, value)


def _lookup_content_id(lookup: Mapping[str, str], name: str) -> str | None:
    for key in _name_keys(name):
        value = lookup.get(key)
        if value is not None:
            return value
    return None


def _name_keys(value: object) -> tuple[str, ...]:
    key = _name_key(value)
    keys = [key]
    if key.startswith("the"):
        keys.append(key[3:])
    else:
        keys.append(f"the{key}")
    return tuple(dict.fromkeys(keys))


def _name_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _is_generic_card_name(name: str) -> bool:
    return _name_key(name) in {
        "card",
        "cards",
        "cardfromyourdeck",
        "cardsfromyourdeck",
    }


def _initial_options(event: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    direct = _mapping_sequence(event.get("options"))
    if direct:
        return direct

    for page in _mapping_sequence(event.get("pages")):
        if _normalized_id(page.get("id", "")) == "initial":
            return _mapping_sequence(page.get("options"))
    return ()


def _has_followup_options(event: Mapping[str, Any], option: Mapping[str, Any]) -> bool:
    option_id = _normalized_id(option.get("id", option.get("title", "")))
    if not option_id:
        return False
    for page in _mapping_sequence(event.get("pages")):
        if _normalized_id(page.get("id", "")) != option_id:
            continue
        return bool(_mapping_sequence(page.get("options")))
    return False


def _load_rows(dataset: str) -> tuple[Mapping[str, Any], ...]:
    path = _CACHE_DIR / f"{dataset}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return _mapping_sequence(payload)


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _str_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value)


def _event_key(event_id: str) -> str:
    key = _normalized_id(event_id)
    return _EVENT_ALIASES.get(key, key)


def _normalized_id(value: object) -> str:
    return str(value).lower().replace("'", "").replace(" ", "_").replace("-", "_")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


_CATALOG, _COVERAGE, _ALL_EVENT_IDS = _build_event_catalog()

__all__ = [
    "EventCatalogOptionCoverage",
    "event_catalog_coverage",
    "event_catalog_ids",
    "event_catalog_unsupported_categories",
    "event_catalog_unsupported_options",
    "known_event_catalog_options",
]

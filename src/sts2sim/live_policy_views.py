"""Visible decision views for live screens; these are not simulator checkpoints.

Only the current enabled controls become candidates. No transition is run and
no hidden RNG, future reward or map encounter is reconstructed. The trained
policy scores the same card/item/action feature vocabulary used in simulation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from sts2sim.agent_api import (
    _aggregate_selected_card_descriptor,
    _card_descriptor,
    action_space,
    encode_observation,
)
from sts2sim.api import serialize
from sts2sim.engine.models import (
    Action,
    ActionType,
    AncientOptionState,
    AncientState,
    CardInstance,
    EventOptionState,
    EventState,
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
    RunState,
    ShopItemState,
    ShopState,
)
from sts2sim.engine.rng import new_rng_state
from sts2sim.live_parity import (
    DEFAULT_CARD_CACHE,
    _card_library,
    _cards_from_live_zone,
    _live_character_id,
    _player_state_from_live,
    _source_card_id_to_runtime_id,
    _source_id_to_runtime_id,
)
from sts2sim.live_relic_state import relic_runtime_projection
from sts2sim.mechanics.relics import relic_potion_slot_bonus
from sts2sim.mechanics.rewards import potion_slots_for_ascension
from sts2sim.mechanics.semantics import action_mechanic_profile
from sts2sim.mechanics.synergy import action_synergy_profile
from sts2sim.policy_visibility import policy_payload


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list | tuple) or any(not isinstance(v, Mapping) for v in value):
        raise ValueError(f"Missing or invalid visible {field_name} list")
    return tuple(value)


def _int(row: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Missing or invalid visible {key} integer")
    return value


def _event_option_enabled(row: Mapping[str, Any]) -> bool:
    for key in ("can_choose", "disable_on_chosen"):
        if key in row and not isinstance(row[key], bool):
            raise ValueError(f"Invalid visible event option {key} boolean")
    if row.get("is_locked") is True or row.get("is_enabled") is False:
        return False
    if "can_choose" in row:
        return row["can_choose"] is True
    # Older captures have no derived selection gate. Chosen options remain
    # blocked unless the source explicitly says choosing does not disable them.
    return row.get("is_enabled") is True and (
        row.get("was_chosen") is not True or row.get("disable_on_chosen") is False
    )


def _key(action: Action) -> str:
    return json.dumps(action.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


@dataclass
class LivePolicyView:
    """A visible feature container and explicit controls, never passed to step()."""

    state: RunState
    choices: list[tuple[Action, dict[str, Any]]] = field(default_factory=list)
    card_features: dict[str, list[CardInstance]] = field(default_factory=dict)
    feature_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    card_zones: dict[str, str] = field(default_factory=dict)

    def add(
        self,
        action: Action,
        command: dict[str, Any],
        *cards: CardInstance,
        features: dict[str, Any] | None = None,
        card_zone: str = "reward",
    ) -> None:
        if any(_key(existing) == _key(action) for existing, _ in self.choices):
            raise ValueError("Duplicate visible policy action identity")
        self.choices.append((action, command))
        if cards:
            self.card_features[_key(action)] = list(cards)
            self.card_zones[_key(action)] = card_zone
        if features:
            self.feature_overrides[_key(action)] = features

    def inputs(
        self, *, agent_memory: object | None = None
    ) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
        actions = [action for action, _ in self.choices]
        if not actions:
            raise ValueError("Live screen has no enabled mapped policy controls")
        descriptors = action_space(self.state, actions=actions)
        visible = policy_payload(serialize(self.state))
        commands_by_key = {_key(action): command for action, command in self.choices}
        commands = {}
        for descriptor in descriptors:
            key = _key(Action.model_validate(descriptor["action"]))
            commands[descriptor["id"]] = commands_by_key[key]
            cards = self.card_features.get(key)
            if cards:
                values = [
                    _card_descriptor(card, zone=self.card_zones[key], position=i)
                    for i, card in enumerate(cards)
                ]
                descriptor["card"] = values[0]
                if len(values) > 1:
                    descriptor["card"] = _aggregate_selected_card_descriptor(values)
                    descriptor["selected_cards"] = values
                    descriptor["selected_card_count"] = len(values)
            descriptor.update(self.feature_overrides.get(key, {}))
            descriptor["mechanics"] = action_mechanic_profile(descriptor)
            descriptor["synergy"] = action_synergy_profile(visible, descriptor)
        observation = encode_observation(self.state, actions=actions, agent_memory=agent_memory)
        observation["action_mask"] = [1] * len(descriptors)
        return (
            observation,
            {"action_space": descriptors, "action_mask": observation["action_mask"]},
            commands,
        )


_ROOMS = {
    "unassigned": RoomKind.EVENT,
    "unknown": RoomKind.EVENT,
    "event": RoomKind.EVENT,
    "start": RoomKind.START,
    "ancient": RoomKind.START,
    "monster": RoomKind.MONSTER,
    "elite": RoomKind.ELITE,
    "boss": RoomKind.BOSS,
    "merchant": RoomKind.SHOP,
    "shop": RoomKind.SHOP,
    "restsite": RoomKind.REST,
    "rest": RoomKind.REST,
    "treasure": RoomKind.TREASURE,
}


def _node_id(row: Mapping[str, Any]) -> str:
    return f"live:{_int(row, 'col')}:{_int(row, 'row')}"


def import_policy_view(
    live: Mapping[str, Any], *, card_cache: Path | str = DEFAULT_CARD_CACHE
) -> LivePolicyView:
    """Describe one current UI decision without inventing a playable save state."""
    player, run = _map(live.get("player")), _map(live.get("run"))
    character = _live_character_id(str(player.get("character_id") or player.get("character") or ""))
    if character is None:
        raise ValueError("Live player is missing a supported character")
    for key in ("hp", "max_hp", "gold", "max_potion_slots"):
        _int(player, key, minimum=1 if key == "max_hp" else 0)
    act = _int(run, "act", minimum=1)
    floor = _int(run, "act_floor") if "act_floor" in run else _int(run, "floor")
    if act > 1 and "act_floor" not in run:
        raise ValueError("Later acts require an explicit within-act floor")
    library = _card_library(Path(card_cache))

    def cards(raw: Any, *, zone: str) -> tuple[CardInstance, ...]:
        rows = _rows(raw, zone)
        for row in rows:
            if (
                not row.get("id")
                or _source_card_id_to_runtime_id(str(row["id"]), tuple(library)) not in library
            ):
                raise ValueError(f"Unknown visible card identity in {zone}")
            if not isinstance(row.get("is_upgraded"), bool):
                raise ValueError("Visible cards require explicit upgrade state")
        return _cards_from_live_zone(
            rows, zone=zone, character_id=character, card_library=library, start_index=1
        )

    deck = cards(player.get("master_deck", player.get("deck")), zone="master")
    relics = tuple(
        _source_id_to_runtime_id(str(row["id"]))
        for row in _rows(player.get("relics"), "relics")
        if row.get("id")
    )
    potions = _rows(player.get("potions"), "potions")
    occupied = [
        (int(row.get("slot", i)), _source_id_to_runtime_id(str(row["id"])))
        for i, row in enumerate(potions)
        if row.get("id")
    ]
    capacity = _int(player, "max_potion_slots")
    if len({slot for slot, _ in occupied}) != len(occupied) or any(
        slot < 0 or slot >= capacity for slot, _ in occupied
    ):
        raise ValueError("Invalid visible physical potion slots")
    ascension = _int(run, "ascension")
    base_capacity = (
        capacity
        + 100
        - potion_slots_for_ascension(100, ascension)
        - relic_potion_slot_bonus(relics)
    )
    if base_capacity < 0:
        raise ValueError("Live capacity contradicts owned relics")
    relic_flags, _ = relic_runtime_projection(player, {})
    state = RunState(
        seed="live-decision-view",
        character_id=character,
        ascension=ascension,
        rng=new_rng_state("unused-live-view"),
        phase=RunPhase.EVENT,
        act=act,
        floor=floor,
        player=_player_state_from_live(player),
        master_deck=deck,
        relics=relics,
        potions=tuple(potion for _, potion in occupied),
        flags={
            "cards": tuple(library.values()),
            **relic_flags,
            "potion_slots": base_capacity,
            "potion_slot_indices": tuple(slot for slot, _ in occupied),
            "can_remove_potions": player.get("can_remove_potions", True),
            "live_observation_only": True,
        },
    )
    view = LivePolicyView(state)
    screen = str(live.get("state_type", "")).lower()
    if screen == "map":
        payload = _map(live.get("map"))
        rows = _rows(payload.get("nodes"), "map.nodes")
        nodes, edges = [], []
        for row in rows:
            kind = _ROOMS.get(str(row.get("type", "")).replace("_", "").lower())
            if kind is None:
                raise ValueError(f"Unknown live map node type: {row.get('type')}")
            nodes.append(
                MapNodeState(
                    node_id=_node_id(row),
                    act=act,
                    floor=_int(row, "row"),
                    lane=_int(row, "col"),
                    kind=kind,
                )
            )
            for child in row.get("children", []):
                if not isinstance(child, list | tuple) or len(child) != 2:
                    raise ValueError("Invalid live map edge")
                edges.append(
                    MapEdgeState(
                        from_id=_node_id(row), to_id=_node_id({"col": child[0], "row": child[1]})
                    )
                )
        current = (
            _node_id(_map(payload["current_position"])) if payload.get("current_position") else None
        )
        game_map = MapState(
            act=act,
            nodes=tuple(nodes),
            edges=tuple(edges),
            current_node_id=current,
            completed_node_ids=tuple(
                _node_id(row) for row in _rows(payload.get("visited", []), "map.visited")
            ),
        )
        view.state = state.model_copy(update={"phase": RunPhase.MAP, "map": game_map})
        for row in _rows(payload.get("next_options"), "map.next_options"):
            if _node_id(row) not in game_map.node_by_id:
                raise ValueError("Travelable map node is missing from visible map")
            view.add(
                Action(type=ActionType.CHOOSE_NODE, target_id=_node_id(row)),
                {"action": "choose_map_node", "index": _int(row, "index")},
            )
    elif screen == "event":
        payload = _map(live.get("event"))
        if payload.get("in_dialogue") is True:
            view.add(
                Action(type=ActionType.PROCEED, target_id="ui:dialogue"),
                {"action": "advance_dialogue"},
            )
        else:
            options, ancient_options = [], []
            for row in _rows(payload.get("options"), "event.options"):
                index = _int(row, "index")
                identity = str(row.get("option_id") or f"live:option:{index}")
                enabled = _event_option_enabled(row)
                options.append(
                    EventOptionState(
                        option_id=identity,
                        title=str(row.get("title", "")),
                        description=str(row.get("description", "")),
                        disabled=not enabled,
                    )
                )
                if not enabled:
                    continue
                action = Action(type=ActionType.CHOOSE_EVENT, target_id=identity)
                if payload.get("is_ancient") is True and row.get("relic_id"):
                    ancient_options.append(
                        AncientOptionState(
                            option_id=identity,
                            name=str(row.get("title", "")),
                            kind="positive_relic",
                            relic_id=_source_id_to_runtime_id(str(row["relic_id"])),
                            description=str(row.get("description", "")),
                        )
                    )
                    action = Action(type=ActionType.CHOOSE_ANCIENT, target_id=identity)
                view.add(action, {"action": "choose_event_option", "index": index})
            if ancient_options:
                view.state = state.model_copy(
                    update={
                        "phase": RunPhase.ANCIENT,
                        "ancient": AncientState(
                            act=act,
                            ancient_id=str(payload.get("event_id", "")).lower(),
                            options=tuple(ancient_options),
                        ),
                    }
                )
            else:
                view.state = state.model_copy(
                    update={
                        "event": EventState(
                            event_id=str(payload.get("event_id", "")).lower(),
                            name=str(payload.get("event_name", "")),
                            page_id=str(payload.get("body_key", "INITIAL")),
                            options=tuple(options),
                        )
                    }
                )
    elif screen == "rest_site":
        payload = _map(live.get("rest_site"))
        view.state = state.model_copy(update={"phase": RunPhase.REST})
        for row in _rows(payload.get("options"), "rest_site.options"):
            if row.get("is_enabled") is not True:
                continue
            identity = str(row.get("id", "")).lower()
            identity = "rest" if identity == "heal" else identity
            try:
                rest_type = ActionType(identity)
            except ValueError as exc:
                raise ValueError(f"Unknown source rest option {identity}") from exc
            if rest_type not in {
                ActionType.REST,
                ActionType.SMITH,
                ActionType.DIG,
                ActionType.LIFT,
                ActionType.KINDLE,
                ActionType.COOK,
                ActionType.CLONE,
                ActionType.HATCH,
                ActionType.MEND,
                ActionType.TOKE,
                ActionType.RECALL,
            }:
                raise ValueError(f"Invalid source rest option {identity}")
            view.add(
                Action(
                    type=rest_type,
                    target_id="ui:rest-selector"
                    if rest_type in {ActionType.SMITH, ActionType.TOKE}
                    else None,
                ),
                {"action": "choose_rest_option", "index": _int(row, "index")},
            )
        if payload.get("can_proceed") is True:
            view.add(Action(type=ActionType.PROCEED), {"action": "proceed"})
    elif screen in {"shop", "fake_merchant"}:
        payload = _map(live.get("shop"))
        if screen == "fake_merchant":
            payload = _map(_map(live.get("fake_merchant")).get("shop")) or payload
        items: list[ShopItemState] = []
        for row in _rows(payload.get("items"), "shop.items"):
            index = _int(row, "index")
            shop_kind = str(row.get("category", ""))
            if shop_kind not in {"card", "colorless_card", "potion", "relic", "card_removal"}:
                raise ValueError(f"Unknown shop entry category: {shop_kind}")
            identifier = str(
                row.get(f"{shop_kind}_id")
                or (row.get("card_id") if shop_kind == "colorless_card" else "")
                or ""
            )
            if shop_kind != "card_removal" and not identifier and row.get("is_stocked") is True:
                raise ValueError("Stocked shop entry is missing its identity")
            card = cards([row["card"]], zone="shop")[0] if row.get("card") else None
            if (
                shop_kind in {"card", "colorless_card"}
                and row.get("is_stocked") is True
                and card is None
            ):
                raise ValueError("Stocked shop card requires its visible card payload")
            item = ShopItemState(
                slot_id=f"shop:{len(items)}",
                item_id=_source_id_to_runtime_id(identifier)
                if shop_kind != "card_removal"
                else "card_removal",
                kind=cast(
                    Literal["card", "colorless_card", "potion", "relic", "card_removal"], shop_kind
                ),
                price=_int(row, "price"),
                base_price=_int(row, "price"),
                purchased=row.get("is_stocked") is not True,
                metadata={
                    "card": card.model_dump() if card else None,
                    "on_sale": row.get("on_sale") is True,
                },
            )
            items.append(item)
            if row.get("is_stocked") is True and row.get("can_afford") is True:
                view.add(
                    Action(type=ActionType.SHOP_BUY, target_id=item.slot_id),
                    {"action": "shop_purchase", "index": index},
                )
        view.state = state.model_copy(
            update={
                "phase": RunPhase.SHOP,
                "shop": ShopState(node_id="live:shop", items=tuple(items)),
            }
        )
        if payload.get("can_proceed") is True:
            view.add(Action(type=ActionType.SHOP_LEAVE), {"action": "proceed"})
    elif screen in {"rewards", "card_reward", "relic_select", "treasure"}:
        payload = _map(live.get(screen))
        view.state = state.model_copy(
            update={"phase": RunPhase.REWARD if screen != "treasure" else RunPhase.TREASURE}
        )
        if screen == "rewards":
            for row in _rows(payload.get("items"), "rewards.items"):
                index = _int(row, "index")
                if row.get("is_enabled") is False:
                    continue
                reward_kind = str(row.get("type", ""))
                if (
                    reward_kind == "card"
                    and "paels_wing" in relics
                    and row.get("selector_can_skip") is True
                    and row.get("selector_can_reroll") is True
                ):
                    # Stable CardRewardAlternative.Generate throws before
                    # opening a selector with more than two alternatives.
                    continue
                target = f"live:reward:{index}"
                kinds = {
                    "gold": ActionType.TAKE_REWARD_GOLD,
                    "relic": ActionType.TAKE_REWARD_RELIC,
                    "potion": ActionType.TAKE_REWARD_POTION,
                    "card": ActionType.CHOOSE_EVENT,
                    "special_card": ActionType.TAKE_REWARD_CARD,
                    "card_removal": ActionType.CHOOSE_EVENT,
                }
                if reward_kind not in kinds:
                    raise ValueError(f"Unknown visible reward entry type: {reward_kind}")
                # An unopened regular card reward is an explicit screen-opening
                # control. Its unshown cards never become policy features.
                card = cards([row["card"]], zone="reward")[0] if row.get("card") else None
                action = Action(type=kinds[reward_kind], target_id=target)
                view.add(
                    action,
                    {"action": "claim_reward", "index": index},
                    *([card] if card else []),
                    features={
                        "reward_choice": {
                            "source": "other",
                            "kind": reward_kind,
                            "content_id": card.card_id
                            if card
                            else _source_id_to_runtime_id(
                                str(row.get(f"{reward_kind}_id", reward_kind))
                            ),
                            "gold": _int(row, "gold_amount") if reward_kind == "gold" else 0,
                            "position": index,
                            "opens_selection": reward_kind in {"card", "card_removal"},
                        },
                        **(
                            {"relic": {"relic_id": _source_id_to_runtime_id(str(row["relic_id"]))}}
                            if reward_kind == "relic" and row.get("relic_id")
                            else {}
                        ),
                        **(
                            {
                                "potion": {
                                    "potion_id": _source_id_to_runtime_id(str(row["potion_id"]))
                                }
                            }
                            if reward_kind == "potion" and row.get("potion_id")
                            else {}
                        ),
                    },
                )
        elif screen == "card_reward":
            offered = cards(payload.get("cards"), zone="reward")
            for row, card in zip(
                _rows(payload.get("cards"), "card_reward.cards"), offered, strict=True
            ):
                view.add(
                    Action(
                        type=ActionType.TAKE_REWARD_CARD,
                        target_id=f"reward:card:{_int(row, 'index')}",
                    ),
                    {"action": "select_card_reward", "card_index": _int(row, "index")},
                    card,
                    features={
                        "reward_choice": {
                            "source": "other",
                            "kind": "card",
                            "content_id": card.card_id,
                            "gold": 0,
                            "position": _int(row, "index"),
                        }
                    },
                )
            for row in _rows(payload.get("alternatives", []), "card_reward.alternatives"):
                if row.get("is_enabled") is not True:
                    continue
                identity = str(row.get("option_id", "")).lower()
                kinds = {
                    "skip": ActionType.SKIP_REWARD,
                    "reroll": ActionType.REROLL_REWARD,
                    "sacrifice": ActionType.SACRIFICE_REWARD,
                    "sacrificecardreward": ActionType.SACRIFICE_REWARD,
                }
                alternative_type = kinds.get(identity)
                if alternative_type is None:
                    raise ValueError(f"Unknown source card reward alternative: {identity}")
                view.add(
                    Action(type=alternative_type, target_id="reward:card_options"),
                    {"action": "skip_card_reward"}
                    if identity == "skip" and payload.get("can_skip") is True
                    else {"action": "select_card_reward_alternative", "index": _int(row, "index")},
                )
            if payload.get("can_skip") is True and not any(
                a.type is ActionType.SKIP_REWARD for a, _ in view.choices
            ):
                view.add(
                    Action(type=ActionType.SKIP_REWARD, target_id="reward:card_options"),
                    {"action": "skip_card_reward"},
                )
        else:
            field_name = "relics"
            for row in _rows(payload.get(field_name, []), f"{screen}.relics"):
                identifier = _source_id_to_runtime_id(str(row.get("id", "")))
                if not identifier:
                    raise ValueError("Visible relic choice is missing identity")
                index = _int(row, "index")
                view.add(
                    Action(
                        type=ActionType.TAKE_REWARD_RELIC,
                        target_id=f"live:relic:{index}",
                        payload={"relic_id": identifier},
                    ),
                    {
                        "action": "select_relic"
                        if screen == "relic_select"
                        else "claim_treasure_relic",
                        "index": index,
                    },
                    features={
                        "relic": {"relic_id": identifier},
                        "reward_choice": {
                            "source": "treasure" if screen == "treasure" else "other",
                            "kind": "relic",
                            "content_id": identifier,
                            "gold": 0,
                            "position": index,
                        },
                    },
                )
            if screen == "relic_select" and payload.get("can_skip") is True:
                view.add(
                    Action(type=ActionType.SKIP_REWARD, target_id="live:relics"),
                    {"action": "skip_relic_selection"},
                )
            if screen == "treasure" and payload.get("can_open_chest") is True:
                view.add(Action(type=ActionType.OPEN_CHEST), {"action": "open_chest"})
        if payload.get("can_proceed") is True:
            view.add(Action(type=ActionType.PROCEED), {"action": "proceed"})
    elif screen in {"card_select", "hand_select", "bundle_select"}:
        payload = _map(live.get(screen))
        if screen == "bundle_select":
            if payload.get("preview_showing") is not True:
                for row in _rows(payload.get("bundles"), "bundle_select.bundles"):
                    bundle = cards(row.get("cards"), zone="bundle")
                    view.add(
                        Action(
                            type=ActionType.CHOOSE_EVENT,
                            target_id=f"live:bundle:{_int(row, 'index')}",
                        ),
                        {"action": "select_bundle", "index": _int(row, "index")},
                        *bundle,
                    )
        elif payload.get("preview_showing") is not True:
            offered = cards(payload.get("cards"), zone="selection")
            for row, card in zip(
                _rows(payload.get("cards"), f"{screen}.cards"), offered, strict=True
            ):
                index = _int(row, "index")
                view.add(
                    Action(type=ActionType.CHOOSE_CARD, card_instance_id=card.instance_id),
                    {
                        "action": "combat_select_card"
                        if screen == "hand_select"
                        else "select_card",
                        "card_index" if screen == "hand_select" else "index": index,
                    },
                    card,
                    card_zone="hand"
                    if screen == "hand_select"
                    else "master_deck"
                    if any(c.instance_id == card.instance_id for c in deck)
                    else "reward",
                )
        if payload.get("can_confirm") is True:
            view.add(
                Action(type=ActionType.PROCEED, target_id="ui:confirm"),
                {
                    "action": "combat_confirm_selection"
                    if screen == "hand_select"
                    else "confirm_bundle_selection"
                    if screen == "bundle_select"
                    else "confirm_selection"
                },
            )
        if payload.get("can_cancel") is True:
            view.add(
                Action(type=ActionType.SKIP_REWARD, target_id="ui:cancel"),
                {
                    "action": "cancel_bundle_selection"
                    if screen == "bundle_select"
                    else "cancel_selection"
                },
            )
    elif screen == "crystal_sphere":
        payload = _map(live.get(screen))
        cells = _rows(payload.get("cells", []), "crystal_sphere.cells")
        visible_cells: list[dict[str, Any]] = [
            {"x": _int(row, "x"), "y": _int(row, "y"), "kind": str(row.get("item_type", "empty"))}
            for row in cells
            if row.get("is_hidden") is False
        ]
        board = {
            "cleared": [row["x"] * 11 + row["y"] for row in visible_cells],
            "visible_cells": visible_cells,
            **(
                {"remaining": _int(payload, "divinations_left")}
                if "divinations_left" in payload
                else {}
            ),
        }
        view.state = state.model_copy(
            update={
                "event": EventState(
                    event_id="crystal_sphere",
                    name="Crystal Sphere",
                    page_id="DIVINATION",
                    metadata={"crystal_sphere_board": board},
                )
            }
        )
        current_tool = str(payload.get("tool", "none"))
        for name in ("big", "small"):
            if payload.get(f"can_use_{name}_tool") is True and name != current_tool:
                view.add(
                    Action(type=ActionType.CHOOSE_EVENT, target_id=f"live:tool:{name}"),
                    {"action": "crystal_sphere_set_tool", "tool": name},
                )
        if current_tool in {"big", "small"}:
            for row in _rows(payload.get("clickable_cells"), "crystal_sphere.clickable_cells"):
                x, y = _int(row, "x"), _int(row, "y")
                view.add(
                    Action(type=ActionType.CHOOSE_EVENT, target_id=f"cell:{x}:{y}"),
                    {"action": "crystal_sphere_click_cell", "x": x, "y": y},
                )
        if payload.get("can_proceed") is True:
            view.add(Action(type=ActionType.PROCEED), {"action": "crystal_sphere_proceed"})
    else:
        raise ValueError(f"No visible policy mapping for live screen {screen or 'unknown'}")
    # Modal selectors own the input surface. Ordinary noncombat screens expose
    # the source popup's observed permission and physical inventory positions.
    if (
        screen
        not in {
            "card_select",
            "hand_select",
            "bundle_select",
            "relic_select",
            "card_reward",
            "crystal_sphere",
        }
        and player.get("can_remove_potions") is True
    ):
        for row in potions:
            if not row.get("id"):
                continue
            slot = _int(row, "slot")
            if row.get("can_discard") is True:
                view.add(
                    Action(type=ActionType.DISCARD_POTION, target_id=f"potion:{slot}"),
                    {"action": "discard_potion", "slot": slot},
                )
            if (
                row.get("can_use_outside_combat") is True
                and row.get("is_queued") is not True
                and row.get("passes_custom_usability_check") is not False
            ):
                potion_id = _source_id_to_runtime_id(str(row["id"]))
                action_type = (
                    ActionType.THROW_POTION_AT_MERCHANT
                    if potion_id == "foul_potion" and screen in {"shop", "fake_merchant"}
                    else ActionType.USE_POTION
                )
                view.add(
                    Action(
                        type=action_type,
                        target_id="player",
                        payload={"potion_id": potion_id, "potion_slot": f"potion:{slot}"},
                    ),
                    {"action": "use_potion", "slot": slot},
                )
    return view

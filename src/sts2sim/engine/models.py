from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ENGINE_SCHEMA_VERSION = 1
PLAYER_TARGET_ID = "player"


class EngineModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class RunPhase(str, Enum):
    ANCIENT = "ancient"
    MAP = "map"
    COMBAT = "combat"
    EVENT = "event"
    SHOP = "shop"
    REST = "rest"
    TREASURE = "treasure"
    REWARD = "reward"
    COMPLETE = "complete"
    FAILED = "failed"


class ActionType(str, Enum):
    CHOOSE_ANCIENT = "choose_ancient"
    CHOOSE_NODE = "choose_node"
    CHOOSE_EVENT = "choose_event"
    PLAY_CARD = "play_card"
    END_TURN = "end_turn"
    REST = "rest"
    SMITH = "smith"
    RECALL = "recall"
    DIG = "dig"
    LIFT = "lift"
    TOKE = "toke"
    TAKE_REWARD_GOLD = "take_reward_gold"
    TAKE_REWARD_RELIC = "take_reward_relic"
    TAKE_REWARD_CARD = "take_reward_card"
    TAKE_REWARD_POTION = "take_reward_potion"
    SKIP_REWARD = "skip_reward"
    SHOP_BUY = "shop_buy"
    SHOP_LEAVE = "shop_leave"
    USE_POTION = "use_potion"
    CHOOSE_CARD = "choose_card"
    DISCARD_CARD = "discard_card"
    EXHAUST_CARD = "exhaust_card"
    DISCARD_POTION = "discard_potion"
    THROW_POTION_AT_MERCHANT = "throw_potion_at_merchant"
    PROCEED = "proceed"


class CardType(str, Enum):
    ATTACK = "attack"
    SKILL = "skill"
    POWER = "power"
    STATUS = "status"
    CURSE = "curse"
    UNKNOWN = "unknown"


class TargetType(str, Enum):
    NONE = "none"
    SELF = "self"
    ENEMY = "enemy"
    ANY = "any"
    ALL_ENEMIES = "all_enemies"


class RoomKind(str, Enum):
    START = "start"
    MONSTER = "monster"
    ELITE = "elite"
    EVENT = "event"
    SHOP = "shop"
    REST = "rest"
    TREASURE = "treasure"
    BOSS = "boss"


class RngState(EngineModel):
    algorithm: Literal["python.random.MT19937"] = "python.random.MT19937"
    version: int
    internal_state: tuple[int, ...]
    gauss_next: float | None = None


class CardEnchantment(EngineModel):
    keyword: str
    amount: int = 0
    source_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CardInstance(EngineModel):
    instance_id: str
    card_id: str
    name: str = ""
    type: CardType = CardType.UNKNOWN
    cost: int | None = 1
    target: TargetType = TargetType.ENEMY
    effects: dict[str, Any] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()
    exhausts: bool = False
    upgraded: bool = False
    enchantments: tuple[CardEnchantment, ...] = ()
    custom: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def default_name(self) -> CardInstance:
        if not self.name:
            object.__setattr__(self, "name", self.card_id)
        return self


class PlayerState(EngineModel):
    hp: int
    max_hp: int
    block: int = 0
    energy: int = 3
    max_energy: int = 3
    gold: int = 0
    statuses: dict[str, int] = Field(default_factory=dict)
    resources: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def clamp_values(self) -> PlayerState:
        object.__setattr__(self, "max_hp", max(1, self.max_hp))
        object.__setattr__(self, "hp", min(max(0, self.hp), self.max_hp))
        object.__setattr__(self, "block", max(0, self.block))
        object.__setattr__(self, "energy", max(0, self.energy))
        object.__setattr__(self, "max_energy", max(0, self.max_energy))
        object.__setattr__(self, "gold", max(0, self.gold))
        resources: dict[str, int] = {}
        for key, value in self.resources.items():
            if isinstance(value, bool):
                continue
            try:
                resources[str(key)] = max(0, int(value))
            except (TypeError, ValueError):
                continue
        object.__setattr__(self, "resources", resources)
        return self


class MonsterState(EngineModel):
    monster_id: str
    name: str = ""
    hp: int
    max_hp: int
    block: int = 0
    intent: str | None = None
    intent_damage: int = 0
    intent_block: int = 0
    move_id: str | None = None
    next_move_id: str | None = None
    hit_count: int = 1
    statuses: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def clamp_values(self) -> MonsterState:
        object.__setattr__(self, "max_hp", max(1, self.max_hp))
        object.__setattr__(self, "hp", min(max(0, self.hp), self.max_hp))
        object.__setattr__(self, "block", max(0, self.block))
        object.__setattr__(self, "intent_damage", max(0, self.intent_damage))
        object.__setattr__(self, "intent_block", max(0, self.intent_block))
        object.__setattr__(self, "hit_count", max(1, self.hit_count))
        if not self.name:
            object.__setattr__(self, "name", self.monster_id)
        return self

    @property
    def alive(self) -> bool:
        return self.hp > 0


class OrbState(EngineModel):
    orb_id: str
    value: int = 0

    @model_validator(mode="after")
    def normalize_values(self) -> OrbState:
        object.__setattr__(
            self,
            "orb_id",
            str(self.orb_id).strip().lower().replace(" ", "_").replace("-", "_"),
        )
        object.__setattr__(self, "value", max(0, int(self.value)))
        return self


class EffectEvent(EngineModel):
    kind: str
    source_id: str | None = None
    target_id: str | None = None
    amount: int | None = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PendingChoiceState(EngineModel):
    choice_id: str
    kind: str
    source_id: str | None = None
    prompt: str = ""
    zone: str = "hand"
    candidate_ids: tuple[str, ...] = ()
    selected_ids: tuple[str, ...] = ()
    min_choices: int = 1
    max_choices: int = 1
    remaining: int = 1
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_choice(self) -> PendingChoiceState:
        object.__setattr__(self, "kind", str(self.kind).strip().lower().replace(" ", "_"))
        object.__setattr__(self, "zone", str(self.zone).strip().lower().replace(" ", "_"))
        object.__setattr__(
            self,
            "candidate_ids",
            tuple(str(item) for item in self.candidate_ids),
        )
        object.__setattr__(
            self,
            "selected_ids",
            tuple(str(item) for item in self.selected_ids),
        )
        max_choices = max(0, int(self.max_choices))
        min_choices = min(max(0, int(self.min_choices)), max_choices)
        remaining = min(max(0, int(self.remaining)), max_choices)
        object.__setattr__(self, "max_choices", max_choices)
        object.__setattr__(self, "min_choices", min_choices)
        object.__setattr__(self, "remaining", remaining)
        return self


class CombatState(EngineModel):
    turn: int = 1
    player: PlayerState
    monsters: tuple[MonsterState, ...] = ()
    draw_pile: tuple[CardInstance, ...] = ()
    discard_pile: tuple[CardInstance, ...] = ()
    exhaust_pile: tuple[CardInstance, ...] = ()
    hand: tuple[CardInstance, ...] = ()
    orbs: tuple[OrbState, ...] = ()
    orb_slots: int = 0
    draw_per_turn: int = 5
    cards_played_this_turn: tuple[str, ...] = ()
    pending_choices: tuple[PendingChoiceState, ...] = ()
    last_events: tuple[EffectEvent, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def clamp_orb_slots(self) -> CombatState:
        object.__setattr__(self, "orb_slots", min(10, max(0, int(self.orb_slots))))
        if len(self.orbs) > self.orb_slots:
            object.__setattr__(self, "orbs", self.orbs[: self.orb_slots])
        return self


class MapNodeState(EngineModel):
    node_id: str
    act: int
    floor: int
    lane: int
    kind: RoomKind


class MapEdgeState(EngineModel):
    from_id: str
    to_id: str


class MapState(EngineModel):
    act: int = 1
    nodes: tuple[MapNodeState, ...] = ()
    edges: tuple[MapEdgeState, ...] = ()
    current_node_id: str | None = None
    completed_node_ids: tuple[str, ...] = ()
    boss_node_id: str | None = None

    @property
    def node_by_id(self) -> dict[str, MapNodeState]:
        return {node.node_id: node for node in self.nodes}

    @property
    def outgoing_by_id(self) -> dict[str, tuple[str, ...]]:
        outgoing: dict[str, list[str]] = {}
        for edge in self.edges:
            outgoing.setdefault(edge.from_id, []).append(edge.to_id)
        return {key: tuple(value) for key, value in outgoing.items()}


class AncientOptionState(EngineModel):
    option_id: str
    name: str
    kind: Literal["positive_relic", "curse_relic"]
    relic_id: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AncientState(EngineModel):
    act: int
    ancient_id: str = "neow"
    options: tuple[AncientOptionState, ...] = ()
    chosen_option_ids: tuple[str, ...] = ()


class ShopItemState(EngineModel):
    slot_id: str
    item_id: str
    kind: Literal["card", "colorless_card", "potion", "relic", "card_removal"]
    rarity: str | None = None
    price: int
    base_price: int
    purchased: bool = False


class ShopState(EngineModel):
    node_id: str
    items: tuple[ShopItemState, ...] = ()
    card_removals_bought: int = 0


class RewardState(EngineModel):
    reward_id: str
    source: Literal["combat", "event", "treasure", "ancient", "other"] = "other"
    forced: bool = False
    gold: int = 0
    gold_claimed: bool = False
    gold_skipped: bool = False
    relic_id: str | None = None
    relic_claimed: bool = False
    relic_skipped: bool = False
    relic_ids: tuple[str, ...] = ()
    claimed_relic_ids: tuple[str, ...] = ()
    skipped_relic_ids: tuple[str, ...] = ()
    card_ids: tuple[str, ...] = ()
    claimed_card_indices: tuple[int, ...] = ()
    skipped_card_indices: tuple[int, ...] = ()
    card_options: tuple[str, ...] = ()
    card_claimed: bool = False
    card_skipped: bool = False
    card_option_groups: tuple[tuple[str, ...], ...] = ()
    claimed_card_option_group_indices: tuple[int, ...] = ()
    skipped_card_option_group_indices: tuple[int, ...] = ()
    potion_id: str | None = None
    potion_claimed: bool = False
    potion_skipped: bool = False
    potion_ids: tuple[str, ...] = ()
    claimed_potion_indices: tuple[int, ...] = ()
    skipped_potion_indices: tuple[int, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def clamp_reward_values(self) -> RewardState:
        object.__setattr__(self, "gold", max(0, self.gold))
        object.__setattr__(
            self,
            "card_options",
            tuple(str(card_id) for card_id in self.card_options),
        )
        object.__setattr__(
            self,
            "card_option_groups",
            tuple(
                tuple(str(card_id) for card_id in group)
                for group in self.card_option_groups
            ),
        )
        object.__setattr__(
            self,
            "relic_ids",
            tuple(str(relic_id) for relic_id in self.relic_ids),
        )
        object.__setattr__(
            self,
            "claimed_relic_ids",
            tuple(str(relic_id) for relic_id in self.claimed_relic_ids),
        )
        object.__setattr__(
            self,
            "skipped_relic_ids",
            tuple(str(relic_id) for relic_id in self.skipped_relic_ids),
        )
        object.__setattr__(
            self,
            "card_ids",
            tuple(str(card_id) for card_id in self.card_ids),
        )
        object.__setattr__(
            self,
            "claimed_card_indices",
            tuple(sorted({max(0, int(index)) for index in self.claimed_card_indices})),
        )
        object.__setattr__(
            self,
            "skipped_card_indices",
            tuple(sorted({max(0, int(index)) for index in self.skipped_card_indices})),
        )
        object.__setattr__(
            self,
            "claimed_card_option_group_indices",
            tuple(
                sorted(
                    {
                        max(0, int(index))
                        for index in self.claimed_card_option_group_indices
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "skipped_card_option_group_indices",
            tuple(
                sorted(
                    {
                        max(0, int(index))
                        for index in self.skipped_card_option_group_indices
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "potion_ids",
            tuple(str(potion_id) for potion_id in self.potion_ids),
        )
        object.__setattr__(
            self,
            "claimed_potion_indices",
            tuple(sorted({max(0, int(index)) for index in self.claimed_potion_indices})),
        )
        object.__setattr__(
            self,
            "skipped_potion_indices",
            tuple(sorted({max(0, int(index)) for index in self.skipped_potion_indices})),
        )
        return self


class EventOptionState(EngineModel):
    option_id: str
    title: str
    description: str = ""
    disabled: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventState(EngineModel):
    event_id: str
    name: str
    page_id: str = "INITIAL"
    options: tuple[EventOptionState, ...] = ()
    resolved_option_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Action(EngineModel):
    type: ActionType
    card_instance_id: str | None = None
    target_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_action_type_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "action_type" in data and "type" not in data:
            data = dict(data)
            data["type"] = data.pop("action_type")
        return data

    @model_validator(mode="after")
    def require_card_for_play(self) -> Action:
        if self.type == ActionType.PLAY_CARD and not self.card_instance_id:
            raise ValueError("play_card actions require card_instance_id")
        if self.type == ActionType.CHOOSE_NODE and not self.target_id:
            raise ValueError("choose_node actions require target_id")
        if self.type == ActionType.CHOOSE_EVENT and not self.target_id:
            raise ValueError("choose_event actions require target_id")
        if self.type == ActionType.CHOOSE_ANCIENT and not self.target_id:
            raise ValueError("choose_ancient actions require target_id")
        if self.type == ActionType.SMITH and not self.target_id:
            raise ValueError("smith actions require target_id")
        if self.type == ActionType.TOKE and not self.target_id:
            raise ValueError("toke actions require target_id")
        if self.type in {
            ActionType.TAKE_REWARD_GOLD,
            ActionType.TAKE_REWARD_RELIC,
            ActionType.TAKE_REWARD_CARD,
            ActionType.TAKE_REWARD_POTION,
            ActionType.SKIP_REWARD,
        } and not self.target_id:
            raise ValueError(f"{self.type.value} actions require target_id")
        if self.type == ActionType.SHOP_BUY and not self.target_id:
            raise ValueError("shop_buy actions require target_id")
        if self.type == ActionType.DISCARD_POTION and not self.target_id:
            raise ValueError("discard_potion actions require target_id")
        if self.type == ActionType.CHOOSE_CARD and not self.card_instance_id:
            raise ValueError("choose_card actions require card_instance_id")
        if self.type == ActionType.DISCARD_CARD and not self.card_instance_id:
            raise ValueError("discard_card actions require card_instance_id")
        if self.type == ActionType.EXHAUST_CARD and not self.card_instance_id:
            raise ValueError("exhaust_card actions require card_instance_id")
        return self


class ReplayEntry(EngineModel):
    step_index: int
    action: Action
    state_hash_before: str
    state_hash_after: str
    events: tuple[EffectEvent, ...] = ()


class RunState(EngineModel):
    schema_version: int = ENGINE_SCHEMA_VERSION
    seed: int | str
    character_id: str
    ascension: int = 0
    rng: RngState
    phase: RunPhase = RunPhase.COMBAT
    act: int = 1
    floor: int = 0
    player: PlayerState
    master_deck: tuple[CardInstance, ...] = ()
    relics: tuple[str, ...] = ()
    curses: tuple[str, ...] = ()
    potions: tuple[str, ...] = ()
    ancient: AncientState | None = None
    event: EventState | None = None
    reward: RewardState | None = None
    shop: ShopState | None = None
    map: MapState | None = None
    combat: CombatState | None = None
    room_history: tuple[str, ...] = ()
    flags: dict[str, Any] = Field(default_factory=dict)
    replay_log: tuple[ReplayEntry, ...] = ()

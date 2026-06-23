"""Stable content identity vocabulary for policy embeddings.

The PPO action descriptors already expose exact card/relic/potion/event IDs in
structured JSON.  This module turns those IDs into stable integer indices so
the neural policy can learn collision-free embeddings instead of relying only
on lossy hash buckets.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

CONTENT_VOCAB_SCHEMA_VERSION = 1
CONTENT_IDENTITY_SLOTS = 8
CONTENT_IDENTITY_EMBED_DIM = 32
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "eng"


@dataclass(frozen=True, slots=True)
class ContentVocab:
    """Stable token/id mapping used by PPO identity embeddings."""

    tokens: tuple[str, ...]
    checksum: str
    schema_version: int = CONTENT_VOCAB_SCHEMA_VERSION

    def __post_init__(self) -> None:
        tokens = tuple(_unique_tokens((PAD_TOKEN, UNKNOWN_TOKEN, *self.tokens)))
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "checksum", self.checksum or _checksum(tokens))

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unknown_id(self) -> int:
        return 1

    def id_for_token(self, token: str) -> int:
        return self.token_to_id.get(token, self.unknown_id)


@cache
def load_content_vocab(cache_dir: str | Path | None = None) -> ContentVocab:
    """Load a deterministic vocabulary from cached Spire Codex content."""

    root = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    tokens: set[str] = set()
    tokens.update(_dataset_id_tokens(root / "cards.json", "card"))
    tokens.update(_dataset_id_tokens(root / "relics.json", "relic"))
    tokens.update(_dataset_id_tokens(root / "potions.json", "potion"))
    tokens.update(_dataset_id_tokens(root / "monsters.json", "monster"))
    tokens.update(_dataset_id_tokens(root / "powers.json", "status"))
    tokens.update(_event_tokens(root / "events.json"))
    tokens.update(_static_option_tokens())
    ordered = tuple(sorted(tokens))
    return ContentVocab(tokens=ordered, checksum=_checksum((PAD_TOKEN, UNKNOWN_TOKEN, *ordered)))


def content_vocab_metadata(vocab: ContentVocab | None = None) -> dict[str, Any]:
    """Return checkpoint/report metadata for the active content vocabulary."""

    active = vocab or load_content_vocab()
    return {
        "content_vocab_schema_version": active.schema_version,
        "content_vocab_size": active.size,
        "content_vocab_checksum": active.checksum,
        "content_identity_slots": CONTENT_IDENTITY_SLOTS,
        "content_identity_embedding_dim": CONTENT_IDENTITY_EMBED_DIM,
    }


def descriptor_identity_tokens(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    """Return ordered stable identity tokens for one action descriptor."""

    tokens: list[str] = []
    card = _mapping(descriptor.get("card"))
    relic = _mapping(descriptor.get("relic"))
    potion = _mapping(descriptor.get("potion"))
    item = _mapping(descriptor.get("item"))
    reward = _mapping(descriptor.get("reward_choice"))
    target = _mapping(descriptor.get("target"))
    event_option = _mapping(descriptor.get("event_option"))
    ancient_option = _mapping(descriptor.get("ancient_option"))

    _append_content_token(tokens, "card", card.get("card_id"))
    _append_content_token(tokens, "relic", relic.get("relic_id"))
    _append_content_token(tokens, "potion", potion.get("potion_id"))
    _append_content_token(tokens, _item_namespace(item), item.get("item_id"))
    _append_content_token(tokens, _reward_namespace(reward), reward.get("content_id"))
    _append_content_token(tokens, "monster", target.get("monster_id"))
    _append_content_token(tokens, "monster", target.get("target_id"))

    event_id = _normalized_id(event_option.get("event_id"))
    page_id = _normalized_id(event_option.get("page_id"))
    option_id = _normalized_id(event_option.get("option_id"))
    if event_id and option_id:
        if page_id:
            tokens.append(f"event_option:{event_id}:{page_id}:{option_id}")
        tokens.append(f"event_option:{event_id}:{option_id}")
        tokens.append(f"event:{event_id}")

    ancient_id = _normalized_id(ancient_option.get("ancient_id"))
    ancient_option_id = _normalized_id(ancient_option.get("option_id"))
    if ancient_id:
        tokens.append(f"ancient:{ancient_id}")
    if ancient_id and ancient_option_id:
        tokens.append(f"ancient_option:{ancient_id}:{ancient_option_id}")
    _append_content_token(tokens, "relic", ancient_option.get("relic_id"))

    option_slot = _mapping(descriptor.get("option_slot"))
    _append_content_token(
        tokens,
        _option_slot_namespace(option_slot),
        option_slot.get("content_id"),
    )

    for content_id in _sequence(reward.get("sibling_content_ids")):
        _append_content_token(tokens, _reward_namespace(reward), content_id)
    for content_id in _sequence(reward.get("available_content_ids")):
        _append_content_token(tokens, _reward_namespace(reward), content_id)

    return tuple(_unique_tokens(tokens))


def descriptor_identity_ids(
    descriptor: Mapping[str, Any],
    *,
    vocab: ContentVocab | None = None,
    slots: int = CONTENT_IDENTITY_SLOTS,
) -> tuple[int, ...]:
    """Return fixed-length vocabulary IDs for a descriptor."""

    active = vocab or load_content_vocab()
    ids = [active.id_for_token(token) for token in descriptor_identity_tokens(descriptor)]
    padded = ids[: max(0, slots)]
    padded.extend(active.pad_id for _index in range(max(0, slots - len(padded))))
    return tuple(padded)


def _dataset_id_tokens(path: Path, namespace: str) -> set[str]:
    tokens: set[str] = set()
    for row in _load_rows(path):
        content_id = _normalized_id(row.get("id"))
        if content_id:
            tokens.add(f"{namespace}:{content_id}")
    return tokens


def _event_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for event in _load_rows(path):
        event_id = _normalized_id(event.get("id"))
        if not event_id:
            continue
        tokens.add(f"event:{event_id}")
        for option in _event_options(event):
            option_id = _normalized_id(option.get("id") or option.get("title"))
            page_id = _normalized_id(option.get("page_id"))
            if not option_id:
                continue
            tokens.add(f"event_option:{event_id}:{option_id}")
            if page_id:
                tokens.add(f"event_option:{event_id}:{page_id}:{option_id}")
    return tokens


def _event_options(event: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for option in _sequence(event.get("options")):
        option_map = _mapping(option)
        if option_map:
            yield option_map
    for page in _sequence(event.get("pages")):
        page_map = _mapping(page)
        page_id = str(page_map.get("id", ""))
        for option in _sequence(page_map.get("options")):
            option_map = dict(_mapping(option))
            if option_map:
                option_map.setdefault("page_id", page_id)
                yield option_map


def _static_option_tokens() -> set[str]:
    tokens: set[str] = set()
    for ancient_id in ("neow", "orobas", "pael", "tezcatara", "nonupeipe", "tanx", "vakuu"):
        tokens.add(f"ancient:{ancient_id}")
    for room_kind in ("monster", "elite", "event", "shop", "rest", "treasure", "boss"):
        tokens.add(f"node_kind:{room_kind}")
    for token in ("gold:gold", "reward:skip", "reward:proceed", "event_option:skip"):
        tokens.add(token)
    return tokens


def _append_content_token(tokens: list[str], namespace: str, value: Any) -> None:
    content_id = _normalized_id(value)
    if namespace and content_id:
        tokens.append(f"{namespace}:{content_id}")


def _item_namespace(item: Mapping[str, Any]) -> str:
    kind = _normalized_id(item.get("kind"))
    if kind in {"card", "colorless_card"}:
        return "card"
    if kind in {"relic", "potion"}:
        return kind
    if kind == "card_removal":
        return "action"
    return "item"


def _reward_namespace(reward: Mapping[str, Any]) -> str:
    kind = _normalized_id(reward.get("kind"))
    if kind in {"card", "fixed_card", "card_group"}:
        return "card"
    if kind in {"relic", "potion"}:
        return kind
    if kind == "gold":
        return "gold"
    if kind in {"skip", "proceed"}:
        return "reward"
    return "reward"


def _option_slot_namespace(option_slot: Mapping[str, Any]) -> str:
    kind = _normalized_id(option_slot.get("kind"))
    if kind in {"card", "fixed_card", "card_group"}:
        return "card"
    if kind in {"relic", "potion", "gold"}:
        return kind
    if kind in {"skip", "proceed"}:
        return "reward"
    return "option"


def _load_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if isinstance(payload, list):
        return tuple(item for item in payload if isinstance(item, Mapping))
    if isinstance(payload, Mapping):
        rows = payload.get("items") or payload.get("data") or payload.get("rows")
        if isinstance(rows, list):
            return tuple(item for item in rows if isinstance(item, Mapping))
    return ()


def _checksum(tokens: Sequence[str]) -> str:
    payload = json.dumps(list(tokens), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _unique_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        normalized = str(token)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique)


def _normalized_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None or isinstance(value, (Mapping, str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    return ()

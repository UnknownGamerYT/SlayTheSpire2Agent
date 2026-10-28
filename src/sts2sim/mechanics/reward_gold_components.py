"""Independent GoldReward choices for engine models and serialized observations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _get(reward: object, key: str, default: Any = None) -> Any:
    return (reward.get(key, default) if isinstance(reward, Mapping)
            else getattr(reward, key, default))


def _metadata(reward: object) -> dict[str, Any]:
    raw = _get(reward, 'metadata', {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def gold_components(reward: object) -> tuple[dict[str, Any], ...]:
    raw = _metadata(reward).get('gold_reward_components')
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(dict(item) for item in raw if isinstance(item, Mapping)
                     and int(item.get('amount', 0)) > 0)
    amount = int(_get(reward, 'gold', 0))
    return ({'amount': amount},) if amount > 0 else ()


def gold_resolved_indices(reward: object, status: str) -> frozenset[int]:
    metadata = _metadata(reward)
    raw = metadata.get(f'{status}_gold_component_indices', ())
    indices = frozenset(int(index) for index in raw)
    if not indices and bool(_get(reward, f'gold_{status}', False)):
        return frozenset(range(len(gold_components(reward))))
    return indices


def available_gold_choices(reward: object) -> tuple[dict[str, Any], ...]:
    components = gold_components(reward)
    resolved = gold_resolved_indices(reward, 'claimed') | gold_resolved_indices(reward, 'skipped')
    multiple = len(components) > 1
    return tuple({
        'kind': 'gold', 'content_id': f'gold:{index}' if multiple else 'gold',
        'target_id': f'reward:gold:{index}' if multiple else 'reward:gold',
        'amount': int(component['amount']),
        'selection_set_id': f'gold:{index}' if multiple else 'gold',
        'selection_set_size': 1, 'position': index,
        'source_id': component.get('source_id'),
        'was_gold_stolen_back': bool(component.get('was_gold_stolen_back', False)),
    } for index, component in enumerate(components) if index not in resolved)


def gold_choice_for_target(reward: object, target_id: str | None) -> dict[str, Any] | None:
    return next((choice for choice in available_gold_choices(reward)
                 if choice['target_id'] == target_id), None)


def resolve_gold_choice(reward: object, target_id: str, *, claimed: bool) -> dict[str, Any]:
    choice = gold_choice_for_target(reward, target_id)
    if choice is None:
        return {}
    status = 'claimed' if claimed else 'skipped'
    if 'gold_reward_components' not in _metadata(reward):
        return {f'gold_{status}': True}
    indices = gold_resolved_indices(reward, status) | {int(choice['position'])}
    metadata = {**_metadata(reward), f'{status}_gold_component_indices': tuple(sorted(indices))}
    return {'metadata': metadata, f'gold_{status}': len(indices) == len(gold_components(reward))}


def gold_component_count(reward: object, status: str) -> int:
    return len(gold_resolved_indices(reward, status))

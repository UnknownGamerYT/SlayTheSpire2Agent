"""Masked PPO trainer for simulator-wide random-seed learning.

The policy scores the currently legal action descriptors instead of emitting a
fixed global action id. That matters because simulator action ids are local to a
state, while card/node descriptors carry reusable semantics.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape as html_escape
from itertools import count
from pathlib import Path
from typing import Any

from sts2sim.api import serialize
from sts2sim.gymnasium_env import Sts2Env
from sts2sim.history import (
    RunHistory,
    append_history_step,
    record_history_step,
    start_run_history,
    write_run_history,
    write_run_history_html,
    write_run_history_map_text,
)
from sts2sim.learning.content_vocab import (
    CONTENT_IDENTITY_EMBED_DIM,
    CONTENT_IDENTITY_SLOTS,
    content_vocab_metadata,
    descriptor_identity_ids,
    load_content_vocab,
)
from sts2sim.learning.models import LearningProgressPoint, LearningRunResult
from sts2sim.learning.progress import (
    with_moving_averages,
    write_learning_progress_data,
)
from sts2sim.learning.rewards import (
    BREAKDOWN_FIELDS,
    DEFAULT_REWARD_CONFIG,
    deck_delta_summary,
    learning_reward,
)
from sts2sim.mechanics.enemy_traits import ENEMY_TRAIT_KEYS, enemy_trait_vector
from sts2sim.mechanics.mechanic_atoms import (
    CARD_SLOT_KEYS,
    STATUS_ATOM_KEYS,
    card_slot_vector,
    status_atom_vector,
)
from sts2sim.mechanics.option_slots import OPTION_SLOT_KEYS, option_slot_vector
from sts2sim.mechanics.planning_context import reward_plan_summary
from sts2sim.mechanics.semantics import MECHANIC_TAG_BUCKETS, MECHANIC_VALUE_KEYS
from sts2sim.mechanics.synergy import (
    SYNERGY_VALUE_KEYS,
)
from sts2sim.mechanics.synergy import (
    profile_value_vector as synergy_value_vector,
)

NETWORK_SCHEMA_VERSION = 5
PLANNING_HEAD_SCHEMA: tuple[str, ...] = (
    "aggression_target",
    "hp_floor",
    "hp_spend_budget",
    "combat_pace",
    "route_preference",
    "potion_policy",
    "reward_pickiness",
    "expected_hp_loss",
    "expected_turns_to_kill",
    "boss_readiness",
)
PLANNING_HEAD_DIM = len(PLANNING_HEAD_SCHEMA)

PPO_TARGET_PRESETS: dict[str, dict[str, int | str | None]] = {
    "act1-boss": {"target_act": 1, "target_floor": 16, "target_phase": None},
    "act2-boss": {"target_act": 2, "target_floor": 15, "target_phase": None},
    "act3-boss": {"target_act": 3, "target_floor": 15, "target_phase": None},
    "game-clear": {"target_act": 4, "target_floor": 0, "target_phase": "complete"},
}
_ACTION_TYPE_COUNT = 32
_CARD_TYPE_IDS = {"attack": 1, "skill": 2, "power": 3, "status": 4, "curse": 5}
_TARGET_TYPE_IDS = {"none": 0, "self": 1, "enemy": 2, "any": 3, "all_enemies": 4}
_ACTION_TARGET_KIND_IDS = {"": 0, "player": 1, "monster": 2, "unknown": 3}
_CARD_ZONE_IDS = {
    "": 0,
    "master_deck": 1,
    "hand": 2,
    "draw_pile": 3,
    "discard_pile": 4,
    "exhaust_pile": 5,
    "reward": 6,
    "shop": 7,
}
_ITEM_KIND_IDS = {
    "": 0,
    "card": 1,
    "colorless_card": 2,
    "potion": 3,
    "relic": 4,
    "card_removal": 5,
}
_RARITY_IDS = {
    "": 0,
    "starter": 1,
    "basic": 2,
    "common": 3,
    "uncommon": 4,
    "rare": 5,
    "shop": 6,
    "event": 7,
    "boss": 8,
    "ancient": 9,
}
_REWARD_SOURCE_IDS = {
    "": 0,
    "combat": 1,
    "event": 2,
    "treasure": 3,
    "ancient": 4,
    "other": 5,
}
_NODE_KIND_IDS = {
    "start": 0,
    "monster": 1,
    "elite": 2,
    "event": 3,
    "shop": 4,
    "rest": 5,
    "treasure": 6,
    "boss": 7,
}
_INTENT_IDS = {
    "": 0,
    "none": 0,
    "attack": 1,
    "attack_defend": 2,
    "attack_buff": 3,
    "attack_debuff": 4,
    "defend": 5,
    "buff": 6,
    "debuff": 7,
    "strong_debuff": 8,
    "sleep": 9,
    "stun": 10,
    "escape": 11,
    "unknown": 12,
}
_CARD_HASH_BUCKETS = 32
_EFFECT_HASH_BUCKETS = 16
_DETAIL_HASH_BUCKETS = 64
_SYNERGY_TAG_BUCKETS = 48
_PATH_PLAN_FEATURE_DIM = 32
_REWARD_BUNDLE_FEATURE_DIM = 21
_POTION_STRATEGY_FEATURE_DIM = 40
_PREVIEW_FEATURE_KEYS = (
    "preview_error",
    "phase_changed",
    "terminal",
    "act_delta",
    "floor_delta",
    "player_hp_delta",
    "player_block_delta",
    "player_energy_delta",
    "player_gold_delta",
    "player_max_hp_delta",
    "deck_count_delta",
    "relic_count_delta",
    "potion_count_delta",
    "target_is_monster",
    "target_hp_delta",
    "target_block_delta",
    "monster_hp_total_delta",
    "monster_block_total_delta",
    "alive_monster_delta",
    "kills",
    "incoming_damage_delta",
    "hand_delta",
    "draw_pile_delta",
    "discard_pile_delta",
    "exhaust_pile_delta",
    "reward_opened",
    "reward_card_count_delta",
    "reward_relic_count_delta",
    "reward_potion_count_delta",
    "reward_gold_delta",
    "shop_available_item_delta",
    "shop_price_total_delta",
    "ended_turn",
    "combat_ended",
    "lookahead_combat",
    "lookahead_combat_ended",
    "end_turn_available",
    "end_turn_preview_error",
    "projected_player_hp_delta_after_end",
    "projected_damage_taken_after_end",
    "enemy_turn_available",
    "enemy_turn_player_hp_delta",
    "enemy_turn_damage_taken",
    "enemy_turn_player_block_delta",
    "enemy_turn_player_status_delta",
    "enemy_turn_monster_hp_delta",
    "enemy_turn_monster_block_delta",
    "enemy_turn_monster_status_delta",
    "enemy_turn_monsters_killed",
    "enemy_turn_retaliation_damage",
    "enemy_turn_retaliation_kills",
    "enemy_turn_poison_damage",
    "enemy_turn_self_damage",
    "enemy_turn_player_damage_events",
    "enemy_turn_monster_attack_events",
    "enemy_turn_block_events",
    "enemy_turn_buff_events",
    "enemy_turn_debuff_events",
    "enemy_turn_next_incoming_damage",
    "enemy_turn_survives",
    "enemy_turn_death_pending",
    "next_turn_number",
    "next_turn_player_hp",
    "next_turn_player_block",
    "next_turn_player_energy",
    "next_turn_hand_count",
    "next_turn_draw_pile_count",
    "next_turn_discard_pile_count",
    "next_turn_exhaust_pile_count",
    "next_turn_incoming_damage",
    "second_turn_legal_action_count",
    "second_turn_previewed_action_count",
    "second_turn_preview_error_count",
    "second_turn_best_damage",
    "second_turn_best_block",
    "second_turn_best_hp_delta",
    "second_turn_kill_available",
    "second_turn_lethal_available",
)
_PREVIEW_FEATURE_DIM = len(_PREVIEW_FEATURE_KEYS)
ACTION_FEATURE_DIM = (
    65
    + _PATH_PLAN_FEATURE_DIM
    + _REWARD_BUNDLE_FEATURE_DIM
    + _POTION_STRATEGY_FEATURE_DIM
    + _PREVIEW_FEATURE_DIM
    + _CARD_HASH_BUCKETS
    + _EFFECT_HASH_BUCKETS
    + _DETAIL_HASH_BUCKETS
    + len(MECHANIC_VALUE_KEYS)
    + MECHANIC_TAG_BUCKETS
    + len(SYNERGY_VALUE_KEYS)
    + _SYNERGY_TAG_BUCKETS
    + len(CARD_SLOT_KEYS)
    + len(STATUS_ATOM_KEYS)
    + len(ENEMY_TRAIT_KEYS)
    + len(OPTION_SLOT_KEYS)
)


@dataclass(frozen=True)
class TrainingTarget:
    """A run target used by PPO curriculum/evaluation."""

    name: str
    target_act: int
    target_floor: int
    target_phase: str | None = None

    def reached(self, observation: Mapping[str, Any]) -> bool:
        if self.target_phase is not None:
            return str(observation.get("phase", "")) == self.target_phase
        act = _lookup_vector_int(observation, "act")
        floor = _lookup_vector_int(observation, "floor")
        return act > self.target_act or (act == self.target_act and floor >= self.target_floor)


@dataclass(frozen=True)
class _ResumeState:
    path: str | None = None
    result: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _Transition:
    observation_vector: tuple[float, ...]
    action_features: tuple[tuple[float, ...], ...]
    action_identity_ids: tuple[tuple[int, ...], ...]
    action_index: int
    old_log_prob: float
    value: float
    planning_targets: tuple[float, ...]
    planning_outputs: tuple[float, ...]
    reward: float
    done: bool
    teacher_action_index: int | None = None


def train_masked_ppo(
    *,
    target: str = "act1-boss",
    max_batches: int = 20,
    until_stopped: bool = False,
    train_runs_per_batch: int = 64,
    train_max_steps: int = 1200,
    eval_runs: int = 32,
    eval_max_steps: int = 1200,
    seed: int | str = "ppo",
    character_id: str = "IRONCLAD",
    ascension: int = 0,
    hidden_size: int = 256,
    hidden_layers: int = 3,
    head_hidden_layers: int = 2,
    activation: str = "silu",
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    planning_coef: float = 0.1,
    teacher_mix: float = 0.0,
    imitation_coef: float = 0.0,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    target_reward: float = 100.0,
    target_eval_successes: int = 1,
    target_consecutive_successes: int = 1,
    target_success_rate: float = 0.0,
    resume: bool = True,
    resume_from_path: Path | str | None = None,
    model_output_path: Path | str | None = Path("checkpoints/masked_ppo_latest.pt"),
    output_path: Path | str | None = Path("reports/masked_ppo_latest.json"),
    progress_output_path: Path | str | None = Path("reports/masked_ppo_progress.json"),
    report_output_path: Path | str | None = Path("reports/masked_ppo_latest.html"),
    progress_window: int = 20,
    device: str = "auto",
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_reporter: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train a masked PPO policy over random simulator seeds."""

    torch, nn, optim = _load_torch()
    torch_device = _resolve_torch_device(torch, device)
    resolved_target = resolve_ppo_target(target)
    success_rate_threshold = _success_rate_threshold(target_success_rate)
    train_rng = random.Random(f"{seed}:ppo-train")
    eval_rng = random.Random(f"{seed}:ppo-eval")
    model_class = _masked_actor_critic_class(nn)
    observation_dim = len(_empty_observation_vector())
    activation_name = _normalize_activation_name(activation)
    content_vocab = load_content_vocab()
    content_metadata = content_vocab_metadata(content_vocab)
    architecture = {
        "observation_dim": observation_dim,
        "action_feature_dim": ACTION_FEATURE_DIM,
        "path_plan_feature_dim": _PATH_PLAN_FEATURE_DIM,
        "reward_bundle_feature_dim": _REWARD_BUNDLE_FEATURE_DIM,
        "potion_strategy_feature_dim": _POTION_STRATEGY_FEATURE_DIM,
        "action_preview_feature_dim": _PREVIEW_FEATURE_DIM,
        "planning_head_schema": list(PLANNING_HEAD_SCHEMA),
        "planning_head_dim": PLANNING_HEAD_DIM,
        "network_schema_version": NETWORK_SCHEMA_VERSION,
        **content_metadata,
        "hidden_size": hidden_size,
        "hidden_layers": max(1, hidden_layers),
        "head_hidden_layers": max(1, head_hidden_layers),
        "activation": activation_name,
        "uses_agent_memory": True,
        "recurrent": False,
    }
    model = model_class(
        observation_dim=observation_dim,
        action_feature_dim=ACTION_FEATURE_DIM,
        content_vocab_size=content_vocab.size,
        content_identity_slots=CONTENT_IDENTITY_SLOTS,
        content_identity_embedding_dim=CONTENT_IDENTITY_EMBED_DIM,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        head_hidden_layers=head_hidden_layers,
        activation=activation_name,
    )
    model.to(torch_device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    resume_state = _load_checkpoint_if_available(
        torch=torch,
        model=model,
        optimizer=optimizer,
        device=torch_device,
        expected_architecture=architecture,
        resume=resume,
        resume_from_path=resume_from_path,
        model_output_path=model_output_path,
    )
    resumed_from = resume_state.path
    previous_result = _mapping(resume_state.result)
    metadata = {
        "algorithm": "masked_action_descriptor_ppo",
        "seed": seed,
        "character_id": character_id,
        "character_name": _character_display_name(character_id),
        "ascension": ascension,
        "target_eval_successes": max(1, target_eval_successes),
        "target_consecutive_successes": max(1, target_consecutive_successes),
        "target_success_rate": success_rate_threshold,
        **architecture,
        "parameter_count": _parameter_count(model),
        **_torch_device_metadata(torch, torch_device, requested_device=device),
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_ratio": clip_ratio,
        "value_coef": value_coef,
        "entropy_coef": entropy_coef,
        "planning_coef": planning_coef,
        "teacher_mix": max(0.0, min(1.0, float(teacher_mix))),
        "imitation_coef": max(0.0, float(imitation_coef)),
        "ppo_epochs": ppo_epochs,
        "minibatch_size": minibatch_size,
        "target_reward": target_reward,
        "until_stopped": until_stopped,
        "reward_schema_version": 5,
        "reward_config": DEFAULT_REWARD_CONFIG.model_dump(mode="json"),
    }
    teacher_agent = (
        _strategic_teacher_agent()
        if max(0.0, float(teacher_mix)) > 0.0 or max(0.0, float(imitation_coef)) > 0.0
        else None
    )

    training_points = list(_resume_progress_points(previous_result, "progress"))
    evaluation_points = list(_resume_progress_points(previous_result, "evaluation_progress"))
    batch_summaries = [
        dict(summary)
        for summary in _sequence(previous_result.get("batch_summaries"))
        if isinstance(summary, Mapping)
    ]
    total_steps = _resume_total_steps(previous_result, training_points)
    total_reward = _resume_total_reward(previous_result, training_points)
    reached_batch = (
        _int(previous_result.get("reached_batch"))
        if previous_result.get("reached_batch") is not None
        else None
    )
    _advance_run_seed_rng(train_rng, len(training_points))
    _advance_run_seed_rng(eval_rng, len(evaluation_points))

    highlight_run_histories = _resume_highlight_run_histories(previous_result)
    previous_batch_count = len(batch_summaries)
    finite_batch_limit = previous_batch_count + max(0, max_batches)
    requested_new_batches: int | None = None if until_stopped else max(0, max_batches)
    batch_limit: int | None = None if until_stopped else finite_batch_limit
    start_batch = previous_batch_count + 1
    batch_indices = (
        count(start_batch)
        if until_stopped
        else range(start_batch, finite_batch_limit + 1)
    )
    _report_training_progress(
        progress_reporter,
        "trainer_start",
        target=resolved_target.__dict__,
        start_batch=start_batch,
        previous_batches=previous_batch_count,
        batch_limit=batch_limit,
        until_stopped=until_stopped,
        resumed_from_path=resumed_from,
        device=str(torch_device),
    )
    for batch_index in batch_indices:
        _report_training_progress(
            progress_reporter,
            "batch_start",
            batch_index=batch_index,
            train_runs_per_batch=max(0, train_runs_per_batch),
            eval_runs=max(0, eval_runs),
            train_max_steps=train_max_steps,
            eval_max_steps=eval_max_steps,
            runs_trained=len(training_points),
        )
        transitions: list[_Transition] = []
        batch_planning_outputs: list[tuple[float, ...]] = []
        for inner_index in range(max(0, train_runs_per_batch)):
            run_index = len(training_points)
            run_seed = _random_run_seed(train_rng)
            run_result, run_transitions = _collect_training_run(
                torch=torch,
                model=model,
                target=resolved_target,
                run_index=run_index,
                seed=run_seed,
                max_steps=train_max_steps,
                character_id=character_id,
                ascension=ascension,
                device=torch_device,
                gamma=gamma,
                target_reward=target_reward,
                teacher_agent=teacher_agent,
                teacher_mix=max(0.0, min(1.0, float(teacher_mix))),
                imitation_enabled=max(0.0, float(imitation_coef)) > 0.0,
            )
            transitions.extend(run_transitions)
            batch_planning_outputs.extend(
                transition.planning_outputs for transition in run_transitions
            )
            training_points.append(_progress_from_run(run_result, "masked_ppo_train"))
            total_steps += run_result.steps_taken
            total_reward += run_result.total_reward
            _report_training_progress(
                progress_reporter,
                "train_run_end",
                batch_index=batch_index,
                run_position=inner_index + 1,
                run_total=max(0, train_runs_per_batch),
                run_index=run_result.run_index,
                seed=run_result.seed,
                steps_taken=run_result.steps_taken,
                total_reward=run_result.total_reward,
                final_act=run_result.final_act,
                final_floor=run_result.final_floor,
                final_phase=run_result.final_phase,
                reached_target=_run_reached_target(run_result, resolved_target),
                failed_to_continue=run_result.failed_to_continue,
                error=run_result.error,
            )

        if transitions:
            _report_training_progress(
                progress_reporter,
                "ppo_update_start",
                batch_index=batch_index,
                transition_count=len(transitions),
            )
            _ppo_update(
                torch=torch,
                model=model,
                optimizer=optimizer,
                transitions=transitions,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_ratio=clip_ratio,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                planning_coef=planning_coef,
                imitation_coef=max(0.0, float(imitation_coef)),
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                device=torch_device,
            )
            _report_training_progress(
                progress_reporter,
                "ppo_update_end",
                batch_index=batch_index,
                transition_count=len(transitions),
            )

        _report_training_progress(
            progress_reporter,
            "eval_start",
            batch_index=batch_index,
            eval_runs=max(0, eval_runs),
        )
        eval_results_list: list[LearningRunResult] = []
        for eval_index in range(max(0, eval_runs)):
            eval_result = _evaluate_one_run(
                torch=torch,
                model=model,
                target=resolved_target,
                run_index=len(evaluation_points) + eval_index,
                seed=_random_run_seed(eval_rng),
                max_steps=eval_max_steps,
                character_id=character_id,
                ascension=ascension,
                device=torch_device,
                include_history=True,
            )
            eval_results_list.append(eval_result)
            _report_training_progress(
                progress_reporter,
                "eval_run_end",
                batch_index=batch_index,
                run_position=eval_index + 1,
                run_total=max(0, eval_runs),
                run_index=eval_result.run_index,
                seed=eval_result.seed,
                steps_taken=eval_result.steps_taken,
                total_reward=eval_result.total_reward,
                final_act=eval_result.final_act,
                final_floor=eval_result.final_floor,
                final_phase=eval_result.final_phase,
                reached_target=_run_reached_target(eval_result, resolved_target),
                failed_to_continue=eval_result.failed_to_continue,
                error=eval_result.error,
            )
        eval_results = tuple(eval_results_list)
        highlight_run_histories = _select_highlight_run_histories(
            eval_results,
            target=resolved_target,
            report_output_path=report_output_path,
            output_path=output_path,
        )
        eval_progress = tuple(
            _progress_from_run(run, "masked_ppo_eval") for run in eval_results
        )
        evaluation_points.extend(eval_progress)
        target_successes = sum(
            1 for run in eval_results if _run_reached_target(run, resolved_target)
        )
        target_success_rate = target_successes / len(eval_results) if eval_results else 0.0
        max_consecutive = max_consecutive_target_successes(eval_results, resolved_target)
        batch_reached = (
            target_successes >= max(1, target_eval_successes)
            and max_consecutive >= max(1, target_consecutive_successes)
            and target_success_rate >= success_rate_threshold
        )
        if batch_reached and reached_batch is None:
            reached_batch = batch_index

        batch_summaries.append(
            _ppo_batch_summary(
                batch_index=batch_index,
                trained_runs_total=len(training_points),
                train_total_steps=total_steps,
                eval_results=eval_results,
                target_successes=target_successes,
                target_success_rate_threshold=success_rate_threshold,
                max_consecutive=max_consecutive,
                reached_target=batch_reached,
                planning_outputs=batch_planning_outputs,
            )
        )
        result = _ppo_result(
            target=resolved_target,
            reached_batch=reached_batch,
            max_batches=max_batches,
            previous_batch_count=previous_batch_count,
            requested_new_batches=requested_new_batches,
            batch_limit=batch_limit,
            until_stopped=until_stopped,
            train_runs_per_batch=train_runs_per_batch,
            total_steps=total_steps,
            total_reward=total_reward,
            resumed_from=resumed_from,
            model_output_path=model_output_path,
            output_path=output_path,
            progress_output_path=progress_output_path,
            report_output_path=report_output_path,
            training_points=training_points,
            evaluation_points=evaluation_points,
            batch_summaries=batch_summaries,
            highlight_run_histories=highlight_run_histories,
            metadata=metadata,
        )
        _persist_ppo(
            torch=torch,
            model=model,
            optimizer=optimizer,
            result=result,
            model_output_path=model_output_path,
            output_path=output_path,
            progress_output_path=progress_output_path,
            report_output_path=report_output_path,
            progress_window=progress_window,
        )
        if progress_callback is not None:
            progress_callback(result)
        _report_training_progress(
            progress_reporter,
            "batch_saved",
            batch_index=batch_index,
            batches_completed=len(batch_summaries),
            runs_trained=len(training_points),
            total_steps=total_steps,
            target_successes=target_successes,
            eval_runs=len(eval_results),
            evaluation_average_reward=_average(run.total_reward for run in eval_results),
            evaluation_average_floor=_average(run.final_floor for run in eval_results),
            evaluation_target_success_rate=(
                target_success_rate
            ),
            target_success_rate_threshold=success_rate_threshold,
            reached_target=batch_reached,
            model_path=str(model_output_path) if model_output_path is not None else None,
            output_path=str(output_path) if output_path is not None else None,
        )
        if batch_reached and not until_stopped:
            return result

    result = _ppo_result(
        target=resolved_target,
        reached_batch=reached_batch,
        max_batches=max_batches,
        previous_batch_count=previous_batch_count,
        requested_new_batches=requested_new_batches,
        batch_limit=batch_limit,
        until_stopped=until_stopped,
        train_runs_per_batch=train_runs_per_batch,
        total_steps=total_steps,
        total_reward=total_reward,
        resumed_from=resumed_from,
        model_output_path=model_output_path,
        output_path=output_path,
        progress_output_path=progress_output_path,
        report_output_path=report_output_path,
        training_points=training_points,
        evaluation_points=evaluation_points,
        batch_summaries=batch_summaries,
        highlight_run_histories=highlight_run_histories,
        metadata=metadata,
    )
    _persist_ppo(
        torch=torch,
        model=model,
        optimizer=optimizer,
        result=result,
        model_output_path=model_output_path,
        output_path=output_path,
        progress_output_path=progress_output_path,
        report_output_path=report_output_path,
        progress_window=progress_window,
    )
    if progress_callback is not None:
        progress_callback(result)
    return result


def _report_training_progress(
    progress_reporter: Callable[[Mapping[str, Any]], None] | None,
    event: str,
    **payload: Any,
) -> None:
    if progress_reporter is None:
        return
    progress_reporter({"event": event, **payload})


def _average(values: Iterable[int | float]) -> float:
    total = 0.0
    count_value = 0
    for value in values:
        total += float(value)
        count_value += 1
    return round(total / count_value, 6) if count_value else 0.0


def resolve_ppo_target(target: str) -> TrainingTarget:
    """Resolve a named PPO curriculum target."""

    normalized = target.strip().lower().replace("_", "-")
    if normalized not in PPO_TARGET_PRESETS:
        valid = ", ".join(sorted(PPO_TARGET_PRESETS))
        raise ValueError(f"Unknown target {target!r}. Valid targets: {valid}.")
    preset = PPO_TARGET_PRESETS[normalized]
    return TrainingTarget(
        name=normalized,
        target_act=_int(preset["target_act"]),
        target_floor=_int(preset["target_floor"]),
        target_phase=(
            str(preset["target_phase"])
            if preset["target_phase"] is not None
            else None
        ),
    )


def max_consecutive_target_successes(
    runs: Sequence[LearningRunResult],
    target: TrainingTarget,
) -> int:
    """Return the longest consecutive target-hit streak in run order."""

    best = 0
    current = 0
    for run in runs:
        if _run_reached_target(run, target):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _success_rate_threshold(value: object) -> float:
    return max(0.0, min(1.0, _float(value)))


def _collect_training_run(
    *,
    torch: Any,
    model: Any,
    target: TrainingTarget,
    run_index: int,
    seed: int,
    max_steps: int,
    character_id: str,
    ascension: int,
    device: Any,
    gamma: float,
    target_reward: float,
    teacher_agent: Any | None = None,
    teacher_mix: float = 0.0,
    imitation_enabled: bool = False,
) -> tuple[LearningRunResult, tuple[_Transition, ...]]:
    env = Sts2Env(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        max_episode_steps=max_steps,
        reward_fn=learning_reward,
        include_serialized_state=False,
    )
    transitions: list[_Transition] = []
    total_reward = 0.0
    steps_taken = 0
    terminated = False
    truncated = False
    error: str | None = None
    failed_to_continue = False
    observation: dict[str, Any] = {}
    reward_breakdown_totals: dict[str, float] = {}
    diagnostics: dict[str, float] = {}
    teacher_rng = random.Random(f"{seed}:teacher-mix")
    try:
        observation, info = env.reset()
        for _step_index in range(max_steps):
            teacher_index = (
                _teacher_action_index(env.state, info, teacher_agent)
                if teacher_agent is not None and env.state is not None
                else None
            )
            forced_index = (
                teacher_index
                if teacher_index is not None
                and teacher_mix > 0.0
                and teacher_rng.random() < teacher_mix
                else None
            )
            decision = _choose_action(
                torch,
                model,
                observation,
                info,
                deterministic=False,
                forced_action_index=forced_index,
                device=device,
            )
            if decision is None:
                failed_to_continue = True
                error = "No legal action id was available before target or terminal state."
                break
            (
                action_id,
                action_index,
                log_prob,
                value,
                action_features,
                action_identity_ids,
                decision_context,
            ) = decision
            action_descriptor = _descriptor_for_id(info, action_id)
            if teacher_index is not None:
                decision_context["teacher_action_index"] = float(teacher_index)
            if forced_index is not None:
                decision_context["teacher_forced"] = 1.0
            env.set_pending_policy_output(decision_context)
            before_state = env.state
            next_observation, reward, terminated, truncated, next_info = env.step(action_id)
            reached = target.reached(next_observation)
            reward_breakdown = _reward_breakdown_with_target(
                next_info,
                target_reward=float(target_reward) if reached else 0.0,
            )
            effective_reward = _float(reward_breakdown.get("total"))
            done = terminated or truncated or reached
            _accumulate_reward_breakdown(reward_breakdown_totals, reward_breakdown)
            transitions.append(
                _Transition(
                    observation_vector=_observation_vector(observation),
                    action_features=action_features,
                    action_identity_ids=action_identity_ids,
                    action_index=action_index,
                    old_log_prob=log_prob,
                    value=value,
                    planning_targets=_planning_targets(observation),
                    planning_outputs=_planning_outputs_tuple(decision_context),
                    reward=effective_reward,
                    done=done,
                    teacher_action_index=teacher_index if imitation_enabled else None,
                )
            )
            _accumulate_run_diagnostics(
                diagnostics,
                before_state=before_state,
                after_state=env.state,
                action_descriptor=action_descriptor,
            )
            total_reward += effective_reward
            steps_taken += 1
            observation = next_observation
            info = next_info
            if done:
                break
    except Exception as exc:  # pragma: no cover - defensive runtime capture
        failed_to_continue = True
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    final_phase = str(observation.get("phase", "unknown"))
    if final_phase in {"complete", "failed"} and error is not None:
        failed_to_continue = False
    return (
        LearningRunResult(
            run_index=run_index,
            seed=seed,
            character_id=character_id,
            ascension=ascension,
            policy="masked_ppo_train",
            steps_taken=steps_taken,
            total_reward=round(total_reward, 6),
            terminated=terminated,
            truncated=truncated,
            final_phase=final_phase,
            final_act=_lookup_vector_int(observation, "act"),
            final_floor=_lookup_vector_int(observation, "floor"),
            error=error,
            failed_to_continue=failed_to_continue,
            reward_breakdown_totals=_rounded_reward_totals(reward_breakdown_totals),
            diagnostics=_final_run_diagnostics(diagnostics, env.state),
            steps=(),
        ),
        tuple(transitions),
    )


def _evaluate_one_run(
    *,
    torch: Any,
    model: Any,
    target: TrainingTarget,
    run_index: int,
    seed: int,
    max_steps: int,
    character_id: str,
    ascension: int,
    device: Any,
    include_history: bool = False,
) -> LearningRunResult:
    env = Sts2Env(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        max_episode_steps=max_steps,
        reward_fn=learning_reward,
        include_serialized_state=False,
    )
    total_reward = 0.0
    steps_taken = 0
    terminated = False
    truncated = False
    error: str | None = None
    failed_to_continue = False
    observation: dict[str, Any] = {}
    history: RunHistory | None = None
    reward_breakdown_totals: dict[str, float] = {}
    diagnostics: dict[str, float] = {}
    try:
        observation, info = env.reset()
        if include_history and env.state is not None:
            history = start_run_history(env.state, policy="masked_ppo_eval")
        for step_index in range(max_steps):
            decision = _choose_action(
                torch,
                model,
                observation,
                info,
                deterministic=True,
                device=device,
            )
            if decision is None:
                failed_to_continue = True
                error = "No legal action id was available before target or terminal state."
                break
            action_id = decision[0]
            before_state = env.state
            action_descriptor = _descriptor_for_id(info, action_id)
            action_payload = _action_for_id(info, action_id)
            env.set_pending_policy_output(decision[6])
            observation, reward, terminated, truncated, info = env.step(action_id)
            reward_breakdown = _reward_breakdown_from_info(info)
            _accumulate_reward_breakdown(reward_breakdown_totals, reward_breakdown)
            _accumulate_run_diagnostics(
                diagnostics,
                before_state=before_state,
                after_state=env.state,
                action_descriptor=action_descriptor,
            )
            if history is not None and before_state is not None and env.state is not None:
                if not action_payload:
                    action_payload = dict(_mapping(info.get("action")))
                history = append_history_step(
                    history,
                    record_history_step(
                        step_index=step_index,
                        before_state=before_state,
                        action=action_payload,
                        after_state=env.state,
                        reward=reward,
                        decision=_history_decision_context(
                            decision,
                            reward_breakdown=reward_breakdown,
                        ),
                    ),
                    env.state,
                )
            total_reward += float(reward)
            steps_taken += 1
            if terminated or truncated or target.reached(observation):
                break
    except Exception as exc:  # pragma: no cover - defensive runtime capture
        failed_to_continue = True
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    final_phase = str(observation.get("phase", "unknown"))
    if final_phase in {"complete", "failed"} and error is not None:
        failed_to_continue = False
    return LearningRunResult(
        run_index=run_index,
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        policy="masked_ppo_eval",
        steps_taken=steps_taken,
        total_reward=round(total_reward, 6),
        terminated=terminated,
        truncated=truncated,
        final_phase=final_phase,
        final_act=_lookup_vector_int(observation, "act"),
        final_floor=_lookup_vector_int(observation, "floor"),
        error=error,
        failed_to_continue=failed_to_continue,
        reward_breakdown_totals=_rounded_reward_totals(reward_breakdown_totals),
        diagnostics=_final_run_diagnostics(diagnostics, env.state),
        history=history.model_dump(mode="json") if history is not None else None,
        steps=(),
    )


def _action_for_id(info: Mapping[str, Any], action_id: int) -> dict[str, Any]:
    for descriptor in _action_space(info):
        if _int(descriptor.get("id")) == int(action_id):
            action = descriptor.get("action")
            return dict(action) if isinstance(action, Mapping) else {}
    return {}


def _descriptor_for_id(info: Mapping[str, Any], action_id: int) -> dict[str, Any]:
    for descriptor in _action_space(info):
        if _int(descriptor.get("id")) == int(action_id):
            return dict(descriptor)
    return {}


def _strategic_teacher_agent() -> Any:
    from sts2sim.agents import StrategicAgent

    return StrategicAgent()


def _teacher_action_index(
    state: Any,
    info: Mapping[str, Any],
    teacher_agent: Any | None,
) -> int | None:
    if teacher_agent is None:
        return None
    try:
        decision = teacher_agent.choose_action(state)
    except Exception:
        return None
    action_id = getattr(decision, "action_id", None)
    if action_id is None:
        return None
    for index, descriptor in enumerate(_action_space(info)):
        if _int(descriptor.get("id")) == int(action_id):
            return index
    return None


def _accumulate_run_diagnostics(
    target: dict[str, float],
    *,
    before_state: Any,
    after_state: Any,
    action_descriptor: Mapping[str, Any],
) -> None:
    before = _state_payload(before_state)
    after = _state_payload(after_state)
    action_type = str(action_descriptor.get("type", ""))
    target["actions"] = target.get("actions", 0.0) + 1.0
    if action_type.startswith("take_reward_"):
        target[action_type] = target.get(action_type, 0.0) + 1.0
    if action_type == "shop_buy":
        kind = str(_mapping(action_descriptor.get("item")).get("kind", ""))
        if kind:
            key = f"shop_buy_{kind}"
            target[key] = target.get(key, 0.0) + 1.0
    if action_type == "skip_reward":
        skip_kind = str(
            _mapping(action_descriptor.get("reward_choice")).get("skip_kind", "unknown")
        )
        key = f"skip_reward_{skip_kind or 'unknown'}"
        target[key] = target.get(key, 0.0) + 1.0
    if action_type == "proceed" and str(before.get("phase", "")) in {"reward", "treasure"}:
        for kind in _available_reward_kinds(before):
            key = f"proceed_with_unclaimed_{kind}"
            target[key] = target.get(key, 0.0) + 1.0
    before_cards = _deck_cards(before)
    after_cards = _deck_cards(after)
    if len(after_cards) > len(before_cards):
        target["deck_cards_added"] = target.get("deck_cards_added", 0.0) + (
            len(after_cards) - len(before_cards)
        )
    if len(after_cards) < len(before_cards):
        target["deck_cards_removed"] = target.get("deck_cards_removed", 0.0) + (
            len(before_cards) - len(after_cards)
        )
    if len(after_cards) != len(before_cards):
        _accumulate_deck_delta_diagnostics(target, before, after)
    before_relics = len(
        _sequence(_mapping(before.get("player")).get("relics", before.get("relics")))
    )
    after_relics = len(
        _sequence(_mapping(after.get("player")).get("relics", after.get("relics")))
    )
    if after_relics > before_relics:
        target["relics_gained"] = target.get("relics_gained", 0.0) + (
            after_relics - before_relics
        )


def _accumulate_deck_delta_diagnostics(
    target: dict[str, float],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    summary = deck_delta_summary(before, after)
    for key in (
        "net_score",
        "capability_delta",
        "category_delta_score",
        "problem_relief_score",
        "pressure_cost",
        "synergy_delta",
        "growth_cost",
    ):
        target[f"deck_delta_{key}"] = target.get(f"deck_delta_{key}", 0.0) + _float(
            summary.get(key)
        )
    target["deck_delta_events"] = target.get("deck_delta_events", 0.0) + 1.0
    if bool(summary.get("growth_blocked")):
        target["deck_growth_blocked"] = target.get("deck_growth_blocked", 0.0) + 1.0
    for key, value in _mapping(summary.get("problem_relief")).items():
        if _float(value) > 0:
            diagnostic_key = f"deck_problem_relief_{key}"
            target[diagnostic_key] = target.get(diagnostic_key, 0.0) + _float(value)
    for key, value in _mapping(summary.get("problems_worsened")).items():
        if _float(value) > 0:
            diagnostic_key = f"deck_problem_worsened_{key}"
            target[diagnostic_key] = target.get(diagnostic_key, 0.0) + _float(value)


def _final_run_diagnostics(values: Mapping[str, Any], state: Any) -> dict[str, float]:
    payload = _state_payload(state)
    player = _mapping(payload.get("player"))
    cards = _deck_cards(payload)
    relics = _sequence(player.get("relics", payload.get("relics")))
    potions = _sequence(player.get("potions", payload.get("potions")))
    result = {str(key): round(_float(value), 6) for key, value in values.items()}
    result["final_deck_size"] = float(len(cards) or _int(player.get("deck_count")))
    result["final_unknown_card_count"] = float(
        sum(1 for card in cards if _normalized_id(_mapping(card).get("type")) == "unknown")
    )
    result["final_relic_count"] = float(len(relics))
    result["final_potion_count"] = float(len(potions))
    result["final_gold"] = float(_int(player.get("gold")))
    return result


def _state_payload(state: Any) -> Mapping[str, Any]:
    if isinstance(state, Mapping):
        return state
    if state is None:
        return {}
    try:
        return serialize(state)
    except Exception:
        return {}


def _deck_cards(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    player = _mapping(payload.get("player"))
    cards = _sequence(payload.get("master_deck"))
    if not cards:
        cards = _sequence(player.get("deck"))
    return tuple(_mapping(card) for card in cards)


def _available_reward_kinds(payload: Mapping[str, Any]) -> tuple[str, ...]:
    summary = _mapping(reward_plan_summary(payload))
    if not _bool(summary.get("reward_open")):
        return ()
    available = _mapping(summary.get("available_counts"))
    kinds: list[str] = []
    if _float(available.get("gold")) > 0:
        kinds.append("gold")
    card_count = (
        _float(available.get("cards"))
        + _float(available.get("card_groups"))
        + _float(available.get("fixed_cards"))
    )
    if card_count > 0:
        kinds.append("card")
    if _float(available.get("card_removals")) > 0:
        kinds.append("card_removal")
    if _float(available.get("relics")) > 0:
        kinds.append("relic")
    if _float(available.get("potions")) > 0:
        kinds.append("potion")
    return tuple(kinds)


def _history_decision_context(
    decision: tuple[
        int,
        int,
        float,
        float,
        tuple[tuple[float, ...], ...],
        tuple[tuple[int, ...], ...],
        dict[str, float],
    ],
    *,
    reward_breakdown: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(decision[6])
    context["action_id"] = decision[0]
    context["action_index"] = decision[1]
    context["value"] = round(float(decision[3]), 6)
    context["legal_action_count"] = len(decision[4])
    for key, value in _mapping(reward_breakdown).items():
        context[f"reward_{key}"] = value
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in context.items()
    }


def _choose_action(
    torch: Any,
    model: Any,
    observation: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    deterministic: bool,
    forced_action_index: int | None = None,
    device: Any,
) -> tuple[
    int,
    int,
    float,
    float,
    tuple[tuple[float, ...], ...],
    tuple[tuple[int, ...], ...],
    dict[str, float],
] | None:
    descriptors = _action_space(info)
    if not descriptors:
        return None
    obs_vector = torch.tensor(
        [_observation_vector(observation)],
        dtype=torch.float32,
        device=device,
    )
    action_features_tuple = tuple(_action_features(descriptor) for descriptor in descriptors)
    action_identity_ids_tuple = tuple(
        descriptor_identity_ids(descriptor) for descriptor in descriptors
    )
    action_tensor = torch.tensor(
        [action_features_tuple],
        dtype=torch.float32,
        device=device,
    )
    identity_tensor = torch.tensor(
        [action_identity_ids_tuple],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        logits, value, planning_outputs = model(obs_vector, action_tensor, identity_tensor)
        logits = logits[0]
        distribution = torch.distributions.Categorical(logits=logits)
        if forced_action_index is not None and 0 <= forced_action_index < len(descriptors):
            action_index_tensor = torch.tensor(int(forced_action_index), device=device)
        else:
            action_index_tensor = torch.argmax(logits) if deterministic else distribution.sample()
        action_index = int(action_index_tensor.item())
        log_prob = float(distribution.log_prob(action_index_tensor).item())
        state_value = float(value[0].item())
        confidence = float(distribution.probs[action_index].item())
        entropy = float(distribution.entropy().item())
        planning_values = planning_outputs[0]
    action_id = _int(descriptors[action_index].get("id"))
    decision_context = {
        "action_index": float(action_index),
        "log_prob": log_prob,
        "value": state_value,
        "confidence": confidence,
        "entropy": entropy,
    }
    for index, key in enumerate(PLANNING_HEAD_SCHEMA):
        decision_context[key] = float(planning_values[index].item())
    return (
        action_id,
        action_index,
        log_prob,
        state_value,
        action_features_tuple,
        action_identity_ids_tuple,
        decision_context,
    )


def _ppo_update(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    transitions: Sequence[_Transition],
    gamma: float,
    gae_lambda: float,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    planning_coef: float,
    imitation_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    device: Any,
) -> None:
    returns, advantages = _returns_and_advantages(
        transitions,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    if not transitions:
        return
    order = list(range(len(transitions)))
    for _epoch in range(max(1, ppo_epochs)):
        random.shuffle(order)
        for start in range(0, len(order), max(1, minibatch_size)):
            indices = order[start : start + max(1, minibatch_size)]
            losses = []
            for index in indices:
                transition = transitions[index]
                obs_tensor = torch.tensor(
                    [transition.observation_vector],
                    dtype=torch.float32,
                    device=device,
                )
                action_tensor = torch.tensor(
                    [transition.action_features],
                    dtype=torch.float32,
                    device=device,
                )
                identity_tensor = torch.tensor(
                    [transition.action_identity_ids],
                    dtype=torch.long,
                    device=device,
                )
                logits, value, planning_outputs = model(
                    obs_tensor,
                    action_tensor,
                    identity_tensor,
                )
                distribution = torch.distributions.Categorical(logits=logits[0])
                action_index = torch.tensor(transition.action_index, device=device)
                log_prob = distribution.log_prob(action_index)
                ratio = torch.exp(
                    log_prob - torch.tensor(transition.old_log_prob, device=device)
                )
                advantage = torch.tensor(
                    float(advantages[index]),
                    dtype=torch.float32,
                    device=device,
                )
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
                policy_loss = -torch.min(unclipped, clipped)
                return_value = torch.tensor(
                    float(returns[index]),
                    dtype=torch.float32,
                    device=device,
                )
                value_loss = torch.square(value[0] - return_value)
                planning_target = torch.tensor(
                    [transition.planning_targets],
                    dtype=torch.float32,
                    device=device,
                )
                planning_loss = torch.mean(torch.square(planning_outputs - planning_target))
                entropy_loss = -distribution.entropy()
                imitation_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if imitation_coef > 0 and transition.teacher_action_index is not None:
                    teacher_index = torch.tensor(
                        int(transition.teacher_action_index),
                        device=device,
                    )
                    imitation_loss = -distribution.log_prob(teacher_index)
                losses.append(
                    policy_loss
                    + value_coef * value_loss
                    + entropy_coef * entropy_loss
                    + planning_coef * planning_loss
                    + imitation_coef * imitation_loss
                )
            if not losses:
                continue
            optimizer.zero_grad()
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()


def _returns_and_advantages(
    transitions: Sequence[_Transition],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    advantages = [0.0 for _transition in transitions]
    returns = [0.0 for _transition in transitions]
    next_value = 0.0
    next_advantage = 0.0
    for index in reversed(range(len(transitions))):
        transition = transitions[index]
        nonterminal = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * next_value * nonterminal - transition.value
        advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[index] = advantage
        returns[index] = advantage + transition.value
        next_value = transition.value
        next_advantage = advantage
    mean = sum(advantages) / max(1, len(advantages))
    variance = sum((value - mean) ** 2 for value in advantages) / max(1, len(advantages))
    std = math.sqrt(variance) or 1.0
    normalized = tuple((value - mean) / std for value in advantages)
    return tuple(returns), normalized


def _masked_actor_critic_class(nn: Any) -> Any:
    class MaskedActorCritic(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            *,
            observation_dim: int,
            action_feature_dim: int,
            content_vocab_size: int = 2,
            content_identity_slots: int = CONTENT_IDENTITY_SLOTS,
            content_identity_embedding_dim: int = CONTENT_IDENTITY_EMBED_DIM,
            hidden_size: int,
            hidden_layers: int,
            head_hidden_layers: int,
            activation: str,
        ) -> None:
            super().__init__()
            self.content_identity_slots = max(0, int(content_identity_slots))
            self.content_identity_embedding_dim = max(0, int(content_identity_embedding_dim))
            self.content_embedding = nn.Embedding(
                max(2, int(content_vocab_size)),
                max(1, self.content_identity_embedding_dim),
            )
            identity_dim = self.content_identity_slots * self.content_identity_embedding_dim
            self.observation_encoder = _mlp(
                nn,
                input_dim=observation_dim,
                hidden_size=hidden_size,
                hidden_layers=hidden_layers,
                activation=activation,
            )
            self.action_encoder = _mlp(
                nn,
                input_dim=action_feature_dim + identity_dim,
                hidden_size=hidden_size,
                hidden_layers=hidden_layers,
                activation=activation,
            )
            self.policy_head = _mlp(
                nn,
                input_dim=hidden_size,
                hidden_size=hidden_size,
                hidden_layers=head_hidden_layers,
                activation=activation,
                output_dim=1,
            )
            self.value_head = _mlp(
                nn,
                input_dim=hidden_size,
                hidden_size=hidden_size,
                hidden_layers=head_hidden_layers,
                activation=activation,
                output_dim=1,
            )
            self.planning_head = _mlp(
                nn,
                input_dim=hidden_size,
                hidden_size=hidden_size,
                hidden_layers=head_hidden_layers,
                activation=activation,
                output_dim=PLANNING_HEAD_DIM,
            )

        def forward(
            self,
            observation: Any,
            action_features: Any,
            action_identity_ids: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            state_hidden = self.observation_encoder(observation)
            if self.content_identity_slots:
                if action_identity_ids is None:
                    action_identity_ids = action_features.new_zeros(
                        (
                            action_features.shape[0],
                            action_features.shape[1],
                            self.content_identity_slots,
                        )
                    ).long()
                action_identity_ids = action_identity_ids.clamp(
                    min=0,
                    max=self.content_embedding.num_embeddings - 1,
                )
                identity_features = self.content_embedding(action_identity_ids).reshape(
                    action_features.shape[0],
                    action_features.shape[1],
                    self.content_identity_slots * self.content_identity_embedding_dim,
                )
                action_input = action_features.new_empty(
                    (
                        action_features.shape[0],
                        action_features.shape[1],
                        action_features.shape[2] + identity_features.shape[2],
                    )
                )
                action_input[..., : action_features.shape[2]] = action_features
                action_input[..., action_features.shape[2] :] = identity_features
            else:
                action_input = action_features
            action_hidden = self.action_encoder(action_input)
            combined = action_hidden + state_hidden.unsqueeze(1)
            logits = self.policy_head(combined).squeeze(-1)
            value = self.value_head(state_hidden).squeeze(-1)
            planning_outputs = self.planning_head(state_hidden).sigmoid()
            return logits, value, planning_outputs

    return MaskedActorCritic


def _mlp(
    nn: Any,
    *,
    input_dim: int,
    hidden_size: int,
    hidden_layers: int,
    activation: str,
    output_dim: int | None = None,
) -> Any:
    layers: list[Any] = []
    in_dim = input_dim
    for _index in range(max(1, hidden_layers)):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(_activation_layer(nn, activation))
        in_dim = hidden_size
    if output_dim is not None:
        layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def _activation_layer(nn: Any, activation: str) -> Any:
    normalized = _normalize_activation_name(activation)
    if normalized == "elu":
        return nn.ELU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {activation}")


def _normalize_activation_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "swish": "silu",
        "silu": "silu",
        "gelu": "gelu",
        "relu": "relu",
        "tanh": "tanh",
        "elu": "elu",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported activation {value!r}. Expected one of: {valid}.")
    return aliases[normalized]


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _action_features(descriptor: Mapping[str, Any]) -> tuple[float, ...]:
    action_type_id = _int(descriptor.get("action_type_id"))
    card = _mapping(descriptor.get("card"))
    node = _mapping(descriptor.get("node"))
    target = _mapping(descriptor.get("target"))
    item = _mapping(descriptor.get("item"))
    potion = _mapping(descriptor.get("potion"))
    relic = _mapping(descriptor.get("relic"))
    reward_choice = _mapping(descriptor.get("reward_choice"))
    reward_bundle = _mapping(descriptor.get("reward_bundle"))
    potion_strategy = _mapping(descriptor.get("potion_strategy"))
    event_option = _mapping(descriptor.get("event_option"))
    ancient_option = _mapping(descriptor.get("ancient_option"))
    mechanics = _mapping(descriptor.get("mechanics"))
    synergy = _mapping(descriptor.get("synergy"))
    preview = _mapping(descriptor.get("preview"))
    effect_amounts = _mapping(card.get("effect_amounts"))
    target_statuses = _mapping(target.get("statuses"))
    path = _mapping(node.get("path"))
    features: list[float] = [
        _scaled(action_type_id, _ACTION_TYPE_COUNT),
        1.0 if card else 0.0,
        1.0 if descriptor.get("target_id") is not None else 0.0,
        _scaled(_CARD_TYPE_IDS.get(str(card.get("type", "")), 0), 8),
        _scaled(_int(card.get("cost")), 5),
        _scaled(_TARGET_TYPE_IDS.get(str(card.get("target", "")), 0), 6),
        1.0 if card.get("upgraded") else 0.0,
        1.0 if card.get("exhausts") else 0.0,
        _scaled(_NODE_KIND_IDS.get(str(node.get("kind", "")), 0), 8),
        _scaled(_int(node.get("act")), 4),
        _scaled(_int(node.get("floor")), 16),
        _scaled(_int(node.get("lane")), 8),
        _scaled(_int(effect_amounts.get("damage")), 80),
        _scaled(_int(effect_amounts.get("block")), 80),
        _scaled(_int(effect_amounts.get("draw")), 10),
        _scaled(_int(effect_amounts.get("energy")), 10),
        _scaled(_int(effect_amounts.get("heal")), 80),
        _scaled(_int(effect_amounts.get("status")), 10),
        1.0 if item else 0.0,
        _scaled(_ITEM_KIND_IDS.get(str(item.get("kind", "")), 0), 8),
        _scaled(_RARITY_IDS.get(str(item.get("rarity", "")), 0), 12),
        _scaled(_int(item.get("price")), 500),
        1.0 if potion else 0.0,
        1.0 if relic else 0.0,
        1.0 if event_option else 0.0,
        1.0 if ancient_option else 0.0,
        _scaled(_REWARD_SOURCE_IDS.get(str(reward_choice.get("source", "")), 0), 8),
        1.0 if reward_choice.get("forced") else 0.0,
        _scaled(_CARD_ZONE_IDS.get(str(card.get("zone", "")), 0), 8),
        _scaled(_int(card.get("position")), 20),
        _scaled(_int(item.get("slot_index")), 20),
        _scaled(_int(potion.get("slot_index")), 8),
        _scaled(_int(reward_choice.get("position")), 16),
        _scaled(_int(event_option.get("position")), 16),
        _scaled(_int(ancient_option.get("position")), 8),
        _scaled(_ACTION_TARGET_KIND_IDS.get(str(target.get("kind", "")), 0), 4),
        _scaled(_int(target.get("position")), 10),
        _scaled(_int(target.get("hp")), 500),
        _scaled(_int(target.get("max_hp")), 500),
        _scaled_fraction(target.get("hp_fraction")),
        _scaled(_int(target.get("block")), 200),
        _scaled(_INTENT_IDS.get(str(target.get("intent", "")), 0), 16),
        _scaled(_int(target.get("intent_damage")), 120),
        _scaled(_int(target.get("intent_block")), 120),
        _scaled(_int(target.get("hit_count")), 10),
        _scaled(_int(target.get("status_total")), 100),
        1.0 if target.get("alive") else 0.0,
        1.0 if str(target.get("kind", "")) == "player" else 0.0,
        _scaled(_int(target_statuses.get("poison")), 100),
        _scaled(_int(target_statuses.get("weak")), 10),
        _scaled(_int(target_statuses.get("vulnerable")), 10),
        _signed_scaled(_float(target_statuses.get("strength")), 20),
        _scaled(_int(path.get("path_count")), 64),
        _scaled(_int(path.get("min_depth")), 16),
        _scaled(_int(path.get("max_depth")), 16),
        _scaled(_float(path.get("avg_depth")), 16),
        _scaled(_int(path.get("max_elites")), 5),
        _scaled(_int(path.get("min_rests")), 5),
        _scaled(_int(path.get("max_rests")), 5),
        _scaled(_int(path.get("max_shops")), 5),
        _scaled(_int(path.get("max_monsters")), 15),
        _scaled(_int(path.get("max_events")), 15),
        _scaled(_int(path.get("max_treasures")), 5),
        1.0 if path.get("has_boss_path") else 0.0,
        1.0 if str(node.get("kind", "")) == "boss" else 0.0,
    ]
    features.extend(_path_plan_features(path))
    features.extend(_reward_bundle_features(reward_bundle, reward_choice))
    features.extend(_potion_strategy_features(potion_strategy))
    features.extend(_preview_features(preview))
    card_hash = _hash_bucket(str(card.get("card_id", "")), _CARD_HASH_BUCKETS)
    features.extend(1.0 if index == card_hash else 0.0 for index in range(_CARD_HASH_BUCKETS))
    effect_buckets = _effect_buckets(card, event_option, ancient_option)
    features.extend(
        1.0 if index in effect_buckets else 0.0 for index in range(_EFFECT_HASH_BUCKETS)
    )
    detail_buckets = _detail_buckets(
        descriptor,
        card,
        target,
        item,
        potion,
        relic,
        reward_choice,
        reward_bundle,
        potion_strategy,
        event_option,
        ancient_option,
        preview,
    )
    features.extend(
        1.0 if index in detail_buckets else 0.0 for index in range(_DETAIL_HASH_BUCKETS)
    )
    mechanic_values = _mapping(mechanics.get("values"))
    features.extend(
        _signed_scaled(_float(mechanic_values.get(key)), 100.0)
        for key in MECHANIC_VALUE_KEYS
    )
    mechanic_tag_buckets = {
        _hash_bucket(str(tag), MECHANIC_TAG_BUCKETS)
        for tag in _sequence(mechanics.get("tags"))
    }
    features.extend(
        1.0 if index in mechanic_tag_buckets else 0.0
        for index in range(MECHANIC_TAG_BUCKETS)
    )
    features.extend(
        _signed_scaled(_float(value), 100.0)
        for value in synergy_value_vector(synergy)
    )
    synergy_tag_buckets = {
        _hash_bucket(str(tag), _SYNERGY_TAG_BUCKETS)
        for tag in _sequence(synergy.get("tags"))
    }
    features.extend(
        1.0 if index in synergy_tag_buckets else 0.0
        for index in range(_SYNERGY_TAG_BUCKETS)
    )
    features.extend(card_slot_vector(card))
    features.extend(status_atom_vector(target_statuses))
    features.extend(enemy_trait_vector(target))
    option_slot = _mapping(descriptor.get("option_slot"))
    features.extend(option_slot_vector(option_slot))
    return tuple(features)


def _effect_buckets(
    card: Mapping[str, Any],
    event_option: Mapping[str, Any],
    ancient_option: Mapping[str, Any],
) -> set[int]:
    effect_keys = [str(effect) for effect in _sequence(card.get("effect_keys"))]
    effect_keys.extend(str(key) for key in _sequence(event_option.get("metadata_keys")))
    effect_keys.extend(str(key) for key in _sequence(ancient_option.get("metadata_keys")))
    return {_hash_bucket(effect, _EFFECT_HASH_BUCKETS) for effect in effect_keys}


def _path_plan_features(path: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        _scaled(_float(path.get("avg_elites")), 5),
        _scaled(_float(path.get("avg_monsters")), 15),
        _scaled(_float(path.get("avg_rests")), 6),
        _scaled(_float(path.get("avg_shops")), 5),
        _scaled(_float(path.get("avg_events")), 10),
        _scaled(_float(path.get("avg_treasures")), 5),
        _scaled(_int(path.get("min_fights")), 20),
        _scaled(_int(path.get("max_fights")), 20),
        _scaled(_float(path.get("avg_fights")), 20),
        _scaled_fraction(path.get("boss_path_fraction")),
        _signed_scaled(_float(path.get("avg_aggression_score")), 20.0),
        _signed_scaled(_float(path.get("max_aggression_score")), 20.0),
        _signed_scaled(_float(path.get("avg_safety_score")), 20.0),
        _signed_scaled(_float(path.get("max_safety_score")), 20.0),
        _scaled(_int(path.get("future_card_reward_groups_min")), 20),
        _scaled(_int(path.get("future_card_reward_groups_max")), 20),
        _scaled(_float(path.get("future_card_reward_groups_avg")), 20),
        _scaled(_int(path.get("future_relic_rewards_min")), 10),
        _scaled(_int(path.get("future_relic_rewards_max")), 10),
        _scaled(_float(path.get("future_relic_rewards_avg")), 10),
        _scaled(_int(path.get("first_rest_depth_min")), 16),
        _scaled(_float(path.get("first_rest_depth_avg")), 16),
        _scaled_fraction(path.get("paths_with_rest_fraction")),
        _scaled(_int(path.get("fights_before_first_rest_min")), 12),
        _scaled(_float(path.get("fights_before_first_rest_avg")), 12),
        _scaled(_int(path.get("elites_before_first_rest_max")), 5),
        _scaled(_float(path.get("upgrade_opportunity_avg")), 6),
        _scaled(_float(path.get("heal_opportunity_avg")), 6),
        _scaled_fraction(path.get("current_hp_fraction")),
        _scaled(_int(path.get("upgradeable_card_count")), 40),
        _signed_scaled(_float(path.get("low_hp_aggression_risk_avg")), 20.0),
        _signed_scaled(_float(path.get("boss_prep_score_avg")), 30.0),
    )


def _reward_bundle_features(
    reward_bundle: Mapping[str, Any],
    reward_choice: Mapping[str, Any],
) -> tuple[float, ...]:
    counts = _mapping(reward_bundle.get("available_counts"))
    claimed = _mapping(reward_bundle.get("claimed_counts"))
    return (
        1.0 if reward_bundle else 0.0,
        1.0 if reward_bundle.get("can_skip") else 0.0,
        1.0 if reward_bundle.get("forced") else 0.0,
        1.0 if reward_choice.get("skips_remaining") else 0.0,
        1.0 if reward_choice.get("skips_selection") else 0.0,
        1.0 if reward_choice.get("closes_selection_set") else 0.0,
        _scaled(_int(counts.get("total")), 24),
        _scaled(_int(counts.get("selection_sets")), 12),
        _scaled(_int(counts.get("cards")), 12),
        _scaled(_int(counts.get("card_groups")), 6),
        _scaled(_int(counts.get("fixed_cards")), 12),
        _scaled(_int(counts.get("relics")), 6),
        _scaled(_int(counts.get("potions")), 6),
        _scaled(_int(counts.get("gold")), 1),
        _scaled(_int(reward_choice.get("available_remaining_count")), 24),
        _scaled(_int(reward_choice.get("selection_set_size")), 12),
        _scaled(_int(reward_choice.get("group_size")), 12),
        _scaled(_int(reward_choice.get("group_index")), 8),
        _scaled(_int(reward_choice.get("card_index")), 8),
        _scaled(_int(claimed.get("card_groups")) + _int(claimed.get("primary_card_group")), 8),
        _scaled(_int(claimed.get("relics")) + _int(claimed.get("potions")), 12),
    )


def _potion_strategy_features(potion_strategy: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        1.0 if potion_strategy else 0.0,
        1.0 if potion_strategy.get("combat_present") else 0.0,
        _scaled_fraction(potion_strategy.get("slot_pressure")),
        _scaled(_int(potion_strategy.get("capacity")), 6),
        _scaled(_int(potion_strategy.get("open_slots")), 6),
        1.0 if potion_strategy.get("belt_full") else 0.0,
        1.0 if potion_strategy.get("frees_slot") else 0.0,
        1.0 if potion_strategy.get("takes_slot") else 0.0,
        1.0 if potion_strategy.get("requires_discard") else 0.0,
        _scaled(_int(potion_strategy.get("damage")), 80),
        _scaled(_int(potion_strategy.get("aoe_damage")), 80),
        _scaled(_int(potion_strategy.get("block")), 120),
        _scaled(_int(potion_strategy.get("draw")), 10),
        _scaled(_int(potion_strategy.get("energy")), 10),
        _scaled(_int(potion_strategy.get("heal")), 120),
        _scaled(_int(potion_strategy.get("max_hp_delta")), 40),
        _scaled(_int(potion_strategy.get("status_enemy")), 20),
        _scaled(_int(potion_strategy.get("status_self")), 20),
        _scaled(_int(potion_strategy.get("poison")), 100),
        _scaled(_int(potion_strategy.get("weak")), 10),
        _scaled(_int(potion_strategy.get("vulnerable")), 10),
        _signed_scaled(_float(potion_strategy.get("strength")), 20.0),
        _signed_scaled(_float(potion_strategy.get("dexterity")), 20.0),
        _signed_scaled(_float(potion_strategy.get("focus")), 20.0),
        _scaled(_int(potion_strategy.get("regen")), 20),
        _scaled(_int(potion_strategy.get("intangible")), 5),
        _scaled(_int(potion_strategy.get("buffer")), 5),
        _scaled(_int(potion_strategy.get("card_generation")), 10),
        _scaled(_int(potion_strategy.get("card_recovery")), 5),
        _scaled(_int(potion_strategy.get("random_card_play")), 10),
        _scaled(_int(potion_strategy.get("free_card_play")), 5),
        _scaled(_int(potion_strategy.get("potion_generation")), 5),
        _scaled(_int(potion_strategy.get("persistent_setup")), 10),
        _scaled(_int(potion_strategy.get("temporary_setup")), 10),
        1.0 if potion_strategy.get("lethal_now") else 0.0,
        1.0 if potion_strategy.get("target_lethal_now") else 0.0,
        _scaled(_int(potion_strategy.get("kills_now")), 5),
        _scaled(_int(potion_strategy.get("damage_prevented_this_turn")), 120),
        1.0 if potion_strategy.get("survival_enabling") else 0.0,
        _scaled_fraction(potion_strategy.get("save_priority")),
    )


def _preview_features(preview: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(_scale_preview_value(key, preview.get(key)) for key in _PREVIEW_FEATURE_KEYS)


def _scale_preview_value(key: str, value: object) -> float:
    if key in {
        "preview_error",
        "phase_changed",
        "terminal",
        "target_is_monster",
        "reward_opened",
        "ended_turn",
        "combat_ended",
        "lookahead_combat",
        "lookahead_combat_ended",
        "end_turn_available",
        "end_turn_preview_error",
        "enemy_turn_available",
        "enemy_turn_survives",
        "enemy_turn_death_pending",
        "second_turn_kill_available",
        "second_turn_lethal_available",
    }:
        return _scaled(_int(value), 1)
    if key in {
        "act_delta",
        "floor_delta",
        "player_energy_delta",
        "deck_count_delta",
        "relic_count_delta",
        "potion_count_delta",
        "alive_monster_delta",
        "kills",
        "hand_delta",
        "draw_pile_delta",
        "discard_pile_delta",
        "exhaust_pile_delta",
        "reward_card_count_delta",
        "reward_relic_count_delta",
        "reward_potion_count_delta",
        "shop_available_item_delta",
        "second_turn_legal_action_count",
        "second_turn_previewed_action_count",
        "second_turn_preview_error_count",
        "enemy_turn_monsters_killed",
        "enemy_turn_retaliation_kills",
        "enemy_turn_player_damage_events",
        "enemy_turn_monster_attack_events",
        "enemy_turn_block_events",
        "enemy_turn_buff_events",
        "enemy_turn_debuff_events",
    }:
        return _signed_scaled(_float(value), 20.0)
    if key in {
        "player_hp_delta",
        "player_block_delta",
        "player_max_hp_delta",
        "target_hp_delta",
        "target_block_delta",
        "monster_hp_total_delta",
        "monster_block_total_delta",
        "incoming_damage_delta",
        "projected_player_hp_delta_after_end",
        "projected_damage_taken_after_end",
        "enemy_turn_player_hp_delta",
        "enemy_turn_damage_taken",
        "enemy_turn_player_block_delta",
        "enemy_turn_player_status_delta",
        "enemy_turn_monster_hp_delta",
        "enemy_turn_monster_block_delta",
        "enemy_turn_monster_status_delta",
        "enemy_turn_retaliation_damage",
        "enemy_turn_poison_damage",
        "enemy_turn_self_damage",
        "enemy_turn_next_incoming_damage",
        "next_turn_incoming_damage",
        "second_turn_best_damage",
        "second_turn_best_block",
        "second_turn_best_hp_delta",
    }:
        return _signed_scaled(_float(value), 300.0)
    if key in {
        "player_gold_delta",
        "reward_gold_delta",
        "shop_price_total_delta",
    }:
        return _signed_scaled(_float(value), 1000.0)
    if key == "next_turn_number":
        return _scaled(_int(value), 20)
    if key == "next_turn_player_hp":
        return _scaled(_int(value), 200)
    if key == "next_turn_player_block":
        return _scaled(_int(value), 300)
    if key == "next_turn_player_energy":
        return _scaled(_int(value), 20)
    if key in {
        "next_turn_hand_count",
        "next_turn_draw_pile_count",
        "next_turn_discard_pile_count",
        "next_turn_exhaust_pile_count",
    }:
        return _scaled(_int(value), 80)
    return _signed_scaled(_float(value), 100.0)


def _detail_buckets(
    descriptor: Mapping[str, Any],
    card: Mapping[str, Any],
    target: Mapping[str, Any],
    item: Mapping[str, Any],
    potion: Mapping[str, Any],
    relic: Mapping[str, Any],
    reward_choice: Mapping[str, Any],
    reward_bundle: Mapping[str, Any],
    potion_strategy: Mapping[str, Any],
    event_option: Mapping[str, Any],
    ancient_option: Mapping[str, Any],
    preview: Mapping[str, Any],
) -> set[int]:
    target_statuses = _mapping(target.get("statuses"))
    path = _mapping(_mapping(descriptor.get("node")).get("path"))
    values = (
        descriptor.get("type"),
        descriptor.get("target_id"),
        card.get("card_id"),
        target.get("kind"),
        target.get("target_id"),
        target.get("monster_id"),
        target.get("source_monster_id"),
        target.get("intent"),
        target.get("move_id"),
        item.get("item_id"),
        potion.get("potion_id"),
        relic.get("relic_id"),
        reward_choice.get("kind"),
        reward_choice.get("content_id"),
        reward_choice.get("selection_set_id"),
        f"reward_skip:{reward_choice.get('skips_remaining', '')}",
        f"reward_skip_selection:{reward_choice.get('skips_selection', '')}",
        f"reward_skip_scope:{reward_choice.get('skip_scope', '')}",
        f"reward_skip_kind:{reward_choice.get('skip_kind', '')}",
        f"reward_closes_set:{reward_choice.get('closes_selection_set', '')}",
        event_option.get("event_id"),
        event_option.get("page_id"),
        event_option.get("option_id"),
        ancient_option.get("ancient_id"),
        ancient_option.get("option_id"),
        ancient_option.get("relic_id"),
        f"card_zone:{card.get('zone', '')}",
        f"card_position:{card.get('position', '')}",
        f"shop_slot:{item.get('slot_index', '')}",
        f"potion_slot:{potion.get('slot_index', '')}",
        f"reward_position:{reward_choice.get('position', '')}",
        f"event_position:{event_option.get('position', '')}",
        f"ancient_position:{ancient_option.get('position', '')}",
        f"path_elites:{path.get('max_elites', '')}",
        f"path_rests:{path.get('max_rests', '')}",
        f"path_shops:{path.get('max_shops', '')}",
        f"path_boss:{path.get('has_boss_path', '')}",
        f"path_aggression:{round(_float(path.get('avg_aggression_score')), 1)}",
        f"path_safety:{round(_float(path.get('avg_safety_score')), 1)}",
        f"path_card_rewards:{round(_float(path.get('future_card_reward_groups_avg')), 1)}",
        f"path_relic_rewards:{round(_float(path.get('future_relic_rewards_avg')), 1)}",
        f"path_first_rest:{path.get('first_rest_depth_min', '')}",
        f"path_fights_before_rest:{round(_float(path.get('fights_before_first_rest_avg')), 1)}",
        f"path_hp_band:{int(_scaled_fraction(path.get('current_hp_fraction')) * 4)}",
        f"path_upgradeable:{min(5, _int(path.get('upgradeable_card_count')))}",
        f"potion_belt_full:{potion_strategy.get('belt_full', '')}",
        f"potion_lethal:{potion_strategy.get('lethal_now', '')}",
        f"potion_prevents_death:{potion_strategy.get('survival_enabling', '')}",
        f"potion_save:{round(_float(potion_strategy.get('save_priority')), 1)}",
        f"potion_slot_pressure:{int(_scaled_fraction(potion_strategy.get('slot_pressure')) * 4)}",
        f"enemy_turn_retaliation:{preview.get('enemy_turn_retaliation_damage', '')}",
        f"enemy_turn_kills:{preview.get('enemy_turn_monsters_killed', '')}",
        f"enemy_turn_death:{preview.get('enemy_turn_death_pending', '')}",
        f"enemy_turn_next_incoming:{preview.get('enemy_turn_next_incoming_damage', '')}",
        f"enemy_turn_buff_events:{preview.get('enemy_turn_buff_events', '')}",
        f"enemy_turn_debuff_events:{preview.get('enemy_turn_debuff_events', '')}",
        f"preview_error:{preview.get('error_type', '')}",
        f"preview_terminal:{preview.get('terminal', '')}",
        f"preview_combat_ended:{preview.get('combat_ended', '')}",
        f"preview_reward_opened:{preview.get('reward_opened', '')}",
        f"preview_lethal:{preview.get('second_turn_lethal_available', '')}",
    )
    bundle_values = tuple(
        f"bundle_choice:{_mapping(choice).get('kind', '')}:"
        f"{_mapping(choice).get('selection_set_id', '')}:"
        f"{_mapping(choice).get('content_id', '')}"
        for choice in _sequence(reward_bundle.get("available_choices"))
    )
    sibling_values = tuple(
        f"reward_sibling:{content_id}"
        for content_id in _sequence(reward_choice.get("sibling_content_ids"))
    )
    potion_role_values = tuple(
        f"potion_role:{role}" for role in _sequence(potion_strategy.get("roles"))
    )
    status_values = tuple(f"target_status:{key}:{value}" for key, value in target_statuses.items())
    selected_cards = tuple(_mapping(card) for card in _sequence(descriptor.get("selected_cards")))
    selected_card_values = tuple(
        "selected_card:"
        f"{index}:{card.get('card_id', '')}:{card.get('zone', '')}:"
        f"{card.get('position', '')}"
        for index, card in enumerate(selected_cards[:8])
    )
    return {
        _hash_bucket(str(value), _DETAIL_HASH_BUCKETS)
        for value in (
            *values,
            *bundle_values,
            *sibling_values,
            *potion_role_values,
            *status_values,
            f"selected_card_count:{len(selected_cards)}",
            *selected_card_values,
        )
        if value not in (None, "")
    }


def _observation_vector(observation: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(_float(value) for value in _sequence(observation.get("vector")))


def _planning_targets(observation: Mapping[str, Any]) -> tuple[float, ...]:
    aggression = _mapping(observation.get("aggression"))
    belief = _mapping(observation.get("belief"))
    player = _mapping(observation.get("player"))
    counts = _mapping(observation.get("counts"))
    combat = _mapping(observation.get("combat"))
    reward = _mapping(observation.get("reward"))

    raw_max_hp = player.get("max_hp")
    max_hp = max(1.0, _float(raw_max_hp if raw_max_hp not in (None, "") else 80.0))
    hp = _float(player.get("hp"))
    hp_fraction = _scaled_fraction(hp / max_hp)
    hp_loss = _float(
        belief.get(
            "likely_damage_taken_after_end_turn",
            max(0.0, _float(combat.get("incoming_damage")) - _float(player.get("block"))),
        )
    )
    raw_turns_to_kill = belief.get("turns_to_kill_estimate")
    turns_to_kill = _float(
        raw_turns_to_kill if raw_turns_to_kill not in (None, "") else 0.0
    )
    route_elites = _float(
        belief.get(
            "route_expected_elites_before_boss",
            aggression.get("future_elite_count", 0.0),
        )
    )
    route_rests = _float(
        belief.get(
            "route_expected_rests_before_boss",
            aggression.get("future_rest_count", 0.0),
        )
    )
    raw_deck_count = counts.get("master_deck")
    deck_count = _float(raw_deck_count if raw_deck_count not in (None, "") else 10.0)
    potion_count = _float(counts.get("potions"))
    reward_cards = _float(reward.get("card_count"))
    raw_survival_margin = belief.get("survival_margin")
    survival_margin = _float(
        raw_survival_margin if raw_survival_margin not in (None, "") else hp - hp_loss
    )
    route_preference = _scaled(route_elites * 1.4 + route_rests * 0.4, 8)
    potion_policy = _scaled_fraction(
        (0.6 if survival_margin <= 0 else 0.0)
        + _scaled(potion_count, 3) * 0.25
        + _float(
            aggression.get("target")
            if aggression.get("target") not in (None, "")
            else 0.5
        )
        * 0.25
    )
    reward_pickiness = _scaled_fraction(
        _scaled(deck_count, 35) * 0.7 + (0.2 if reward_cards else 0.0)
    )
    boss_readiness = _scaled_fraction(
        hp_fraction * 0.45
        + _scaled(_float(counts.get("relics")), 18) * 0.25
        + _scaled(potion_count, 3) * 0.15
        + _scaled(_float(belief.get("reward_relic_ev")), 1) * 0.15
    )
    pace = str(aggression.get("combat_pace", "balanced"))
    combat_pace = {"stall": 0.0, "balanced": 0.5, "rush": 1.0}.get(
        pace,
        _scaled_fraction(aggression.get("combat_pace_pressure")),
    )
    values = {
        "aggression_target": _scaled_fraction(aggression.get("target")),
        "hp_floor": _scaled_fraction(aggression.get("hp_floor")),
        "hp_spend_budget": _scaled(_float(aggression.get("hp_spend_budget")), max_hp),
        "combat_pace": combat_pace,
        "route_preference": route_preference,
        "potion_policy": potion_policy,
        "reward_pickiness": reward_pickiness,
        "expected_hp_loss": _scaled(hp_loss, max_hp),
        "expected_turns_to_kill": _scaled(turns_to_kill, 12),
        "boss_readiness": boss_readiness,
    }
    return tuple(float(values[key]) for key in PLANNING_HEAD_SCHEMA)


def _planning_outputs_tuple(payload: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(_scaled_fraction(payload.get(key)) for key in PLANNING_HEAD_SCHEMA)


def _empty_observation_vector() -> tuple[float, ...]:
    env = Sts2Env(seed=0, character_id="TEST", ascension=0, include_serialized_state=False)
    observation, _info = env.reset()
    env.close()
    return _observation_vector(observation)


def _run_reached_target(run: LearningRunResult, target: TrainingTarget) -> bool:
    if target.target_phase is not None:
        return run.final_phase == target.target_phase
    if run.final_act > target.target_act:
        return True
    return run.final_act == target.target_act and run.final_floor >= target.target_floor


def _progress_from_run(run: LearningRunResult, policy: str) -> LearningProgressPoint:
    return LearningProgressPoint(
        run_index=run.run_index,
        seed=run.seed,
        policy=policy,
        steps_taken=run.steps_taken,
        total_reward=run.total_reward,
        final_phase=run.final_phase,
        final_act=run.final_act,
        final_floor=run.final_floor,
        win=run.final_phase == "complete",
        death=run.final_phase == "failed",
        truncated=run.truncated and not run.terminated,
        failed_to_continue=run.failed_to_continue,
        error=run.error,
    )


def _ppo_batch_summary(
    *,
    batch_index: int,
    trained_runs_total: int,
    train_total_steps: int,
    eval_results: Sequence[LearningRunResult],
    target_successes: int,
    target_success_rate_threshold: float,
    max_consecutive: int,
    reached_target: bool,
    planning_outputs: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    completed = len(eval_results)
    planning_averages = _planning_output_averages(planning_outputs)
    reward_averages = _reward_component_averages(eval_results)
    diagnostic_averages = _run_diagnostic_averages(eval_results)
    return {
        "batch_index": batch_index,
        "trained_runs_total": trained_runs_total,
        "train_total_steps": train_total_steps,
        "evaluation_runs": completed,
        "evaluation_average_reward": round(
            sum(run.total_reward for run in eval_results) / max(1, completed),
            6,
        ),
        "evaluation_average_floor": round(
            sum(run.final_floor for run in eval_results) / max(1, completed),
            6,
        ),
        "evaluation_best_floor": max((run.final_floor for run in eval_results), default=0),
        "evaluation_best_reward": round(
            max((run.total_reward for run in eval_results), default=0.0),
            6,
        ),
        "evaluation_target_successes": target_successes,
        "evaluation_target_success_rate": round(target_successes / max(1, completed), 6),
        "target_success_rate_threshold": round(target_success_rate_threshold, 6),
        "evaluation_max_consecutive_successes": max_consecutive,
        "evaluation_errors": sum(1 for run in eval_results if run.error is not None),
        "evaluation_failed_to_continue": sum(
            1 for run in eval_results if run.failed_to_continue
        ),
        "planning_output_averages": planning_averages,
        "reward_component_averages": reward_averages,
        "diagnostic_averages": diagnostic_averages,
        "reached_target": reached_target,
    }


def _planning_output_averages(values: Sequence[Sequence[float]]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in PLANNING_HEAD_SCHEMA}
    return {
        key: round(
            sum(_float(row[index]) for row in values if index < len(row)) / max(1, len(values)),
            6,
        )
        for index, key in enumerate(PLANNING_HEAD_SCHEMA)
    }


def _reward_breakdown_from_info(info: Mapping[str, Any]) -> dict[str, float]:
    raw = _mapping(info.get("reward_breakdown"))
    return {
        key: round(_float(raw.get(key)), 6)
        for key in BREAKDOWN_FIELDS
        if key in raw or key == "total"
    }


def _reward_breakdown_with_target(
    info: Mapping[str, Any],
    *,
    target_reward: float,
) -> dict[str, float]:
    breakdown = _reward_breakdown_from_info(info)
    if target_reward:
        breakdown["target_reached_reward"] = round(
            _float(breakdown.get("target_reached_reward")) + target_reward,
            6,
        )
        breakdown["total"] = round(_float(breakdown.get("total")) + target_reward, 6)
    return breakdown


def _accumulate_reward_breakdown(
    target: dict[str, float],
    breakdown: Mapping[str, Any],
) -> None:
    for key, value in breakdown.items():
        if key == "aggression_pressure":
            target["aggression_pressure_sum"] = target.get(
                "aggression_pressure_sum",
                0.0,
            ) + _float(value)
            target["aggression_pressure_count"] = target.get(
                "aggression_pressure_count",
                0.0,
            ) + 1.0
            continue
        target[str(key)] = target.get(str(key), 0.0) + _float(value)


def _rounded_reward_totals(values: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): round(_float(value), 6) for key, value in values.items()}


def _reward_component_averages(runs: Sequence[LearningRunResult]) -> dict[str, float]:
    if not runs:
        return {key: 0.0 for key in BREAKDOWN_FIELDS if key != "aggression_pressure"}
    keys = {
        key
        for run in runs
        for key in run.reward_breakdown_totals
        if key not in {"aggression_pressure_sum", "aggression_pressure_count"}
    }
    keys.update(key for key in BREAKDOWN_FIELDS if key != "aggression_pressure")
    averages = {
        key: round(
            sum(_float(run.reward_breakdown_totals.get(key)) for run in runs)
            / max(1, len(runs)),
            6,
        )
        for key in sorted(keys)
    }
    pressure_sum = sum(
        _float(run.reward_breakdown_totals.get("aggression_pressure_sum")) for run in runs
    )
    pressure_count = sum(
        _float(run.reward_breakdown_totals.get("aggression_pressure_count")) for run in runs
    )
    averages["aggression_pressure"] = round(pressure_sum / max(1.0, pressure_count), 6)
    return averages


def _run_diagnostic_averages(runs: Sequence[LearningRunResult]) -> dict[str, float]:
    if not runs:
        return {}
    keys = {key for run in runs for key in run.diagnostics}
    return {
        key: round(
            sum(_float(run.diagnostics.get(key)) for run in runs) / max(1, len(runs)),
            6,
        )
        for key in sorted(keys)
    }


def _select_highlight_run_histories(
    eval_results: Sequence[LearningRunResult],
    *,
    target: TrainingTarget,
    report_output_path: Path | str | None,
    output_path: Path | str | None,
) -> dict[str, Any]:
    runs_with_history = tuple(run for run in eval_results if run.history is not None)
    if not runs_with_history:
        return {}
    best = max(runs_with_history, key=lambda run: _highlight_quality_key(run, target))
    worst = min(runs_with_history, key=lambda run: _highlight_quality_key(run, target))
    base = _highlight_artifact_base(report_output_path, output_path)
    generated_at = _utc_timestamp()
    return {
        "schema_version": 2,
        "generated_at": generated_at,
        "best": _highlight_history_entry(
            "best",
            best,
            target=target,
            base=base,
            generated_at=generated_at,
        ),
        "worst": _highlight_history_entry(
            "worst",
            worst,
            target=target,
            base=base,
            generated_at=generated_at,
        ),
    }


def _resume_highlight_run_histories(previous_result: Mapping[str, Any]) -> dict[str, Any]:
    payload = previous_result.get("highlight_run_histories")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _highlight_quality_key(
    run: LearningRunResult,
    target: TrainingTarget,
) -> tuple[int, int, int, int, float, int, int]:
    return (
        int(not run.failed_to_continue and run.error is None),
        int(_run_reached_target(run, target)),
        run.final_act,
        run.final_floor,
        run.total_reward,
        run.steps_taken,
        -run.run_index,
    )


def _highlight_history_entry(
    role: str,
    run: LearningRunResult,
    *,
    target: TrainingTarget,
    base: Path | None,
    generated_at: str,
) -> dict[str, Any]:
    paths = _highlight_paths(base, role) if base is not None else {}
    history = dict(run.history or {})
    history["generated_at"] = generated_at
    history["highlight_role"] = role
    return {
        "role": role,
        "generated_at": generated_at,
        "run_index": run.run_index,
        "seed": run.seed,
        "character_id": run.character_id,
        "ascension": run.ascension,
        "steps_taken": run.steps_taken,
        "total_reward": run.total_reward,
        "terminated": run.terminated,
        "truncated": run.truncated,
        "final_phase": run.final_phase,
        "final_act": run.final_act,
        "final_floor": run.final_floor,
        "target_reached": _run_reached_target(run, target),
        "failed_to_continue": run.failed_to_continue,
        "error": run.error,
        "json_path": str(paths["json"]) if "json" in paths else None,
        "html_path": str(paths["html"]) if "html" in paths else None,
        "map_path": str(paths["map"]) if "map" in paths else None,
        "history": history,
    }


def _highlight_artifact_base(
    report_output_path: Path | str | None,
    output_path: Path | str | None,
) -> Path | None:
    source = report_output_path if report_output_path is not None else output_path
    if source is None:
        return None
    target = Path(source)
    return target.parent / target.stem


def _highlight_paths(base: Path, role: str) -> dict[str, Path]:
    safe_role = role.lower().replace(" ", "_")
    return {
        "json": base.with_name(f"{base.name}_{safe_role}_run_history.json"),
        "html": base.with_name(f"{base.name}_{safe_role}_run_history.html"),
        "map": base.with_name(f"{base.name}_{safe_role}_run_map.txt"),
    }


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ppo_result(
    *,
    target: TrainingTarget,
    reached_batch: int | None,
    max_batches: int,
    previous_batch_count: int,
    requested_new_batches: int | None,
    batch_limit: int | None,
    until_stopped: bool,
    train_runs_per_batch: int,
    total_steps: int,
    total_reward: float,
    resumed_from: str | None,
    model_output_path: Path | str | None,
    output_path: Path | str | None,
    progress_output_path: Path | str | None,
    report_output_path: Path | str | None,
    training_points: Sequence[LearningProgressPoint],
    evaluation_points: Sequence[LearningProgressPoint],
    batch_summaries: Sequence[Mapping[str, Any]],
    highlight_run_histories: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    completed_runs = max(1, len(training_points))
    progress = with_moving_averages(training_points, window=20)
    eval_progress = with_moving_averages(evaluation_points, window=20)
    return {
        "algorithm": "masked_action_descriptor_ppo",
        "target": target.__dict__,
        "reached_target": reached_batch is not None,
        "reached_batch": reached_batch,
        "batches_completed": len(batch_summaries),
        "max_batches": max_batches,
        "until_stopped": until_stopped,
        "previous_batches": previous_batch_count,
        "requested_new_batches": requested_new_batches,
        "batch_limit": batch_limit,
        "train_runs_per_batch": train_runs_per_batch,
        "runs_trained": len(training_points),
        "total_steps": total_steps,
        "average_training_reward": round(total_reward / completed_runs, 6),
        "wins": sum(1 for point in training_points if point.win),
        "deaths": sum(1 for point in training_points if point.death),
        "resumed_from_path": resumed_from,
        "model_path": str(model_output_path) if model_output_path is not None else None,
        "output_path": str(output_path) if output_path is not None else None,
        "progress_output_path": (
            str(progress_output_path) if progress_output_path is not None else None
        ),
        "report_output_path": (
            str(report_output_path) if report_output_path is not None else None
        ),
        "batch_summaries": list(batch_summaries),
        "progress": [point.model_dump(mode="json") for point in progress],
        "evaluation_progress": [point.model_dump(mode="json") for point in eval_progress],
        "highlight_run_histories": dict(highlight_run_histories),
        "metadata": dict(metadata),
    }


def _persist_ppo(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    result: Mapping[str, Any],
    model_output_path: Path | str | None,
    output_path: Path | str | None,
    progress_output_path: Path | str | None,
    report_output_path: Path | str | None,
    progress_window: int,
) -> None:
    if model_output_path is not None:
        target = Path(model_output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "architecture": _checkpoint_architecture(result),
                "result": dict(result),
            },
            target,
        )
    if output_path is not None:
        _write_json(result, output_path)
    _write_highlight_run_history_artifacts(result)
    progress = [
        LearningProgressPoint.model_validate(point)
        for point in result.get("progress", [])
        if isinstance(point, Mapping)
    ]
    if progress_output_path is not None:
        write_learning_progress_data(
            progress,
            progress_output_path,
            title="Masked PPO Training Progress",
            window=progress_window,
        )
    if report_output_path is not None:
        _write_ppo_progress_report(
            result,
            report_output_path,
            progress=progress,
            window=progress_window,
        )


def _write_highlight_run_history_artifacts(result: Mapping[str, Any]) -> None:
    histories = _mapping(result.get("highlight_run_histories"))
    for role in ("best", "worst"):
        entry = _mapping(histories.get(role))
        history = _mapping(entry.get("history"))
        if not history:
            continue
        history = dict(history)
        if entry.get("generated_at"):
            history.setdefault("generated_at", entry.get("generated_at"))
        history.setdefault("highlight_role", role)
        json_path = _optional_path(entry.get("json_path"))
        html_path = _optional_path(entry.get("html_path"))
        map_path = _optional_path(entry.get("map_path"))
        if json_path is not None:
            write_run_history(history, json_path)
        if html_path is not None:
            title = (
                f"{role.title()} PPO Evaluation Run "
                f"(seed {entry.get('seed', '')}, reward {entry.get('total_reward', 0)})"
            )
            write_run_history_html(history, html_path)
            html_text = html_path.read_text(encoding="utf-8")
            title_tag = f"<title>{html_escape(title)}</title>"
            heading = f"<h1>{html_escape(title)}</h1>"
            html_path.write_text(
                html_text.replace("<title>Run History</title>", title_tag).replace(
                    "<h1>Run History</h1>",
                    heading,
                ),
                encoding="utf-8",
            )
        if map_path is not None:
            write_run_history_map_text(history, map_path)


def _highlight_links_html(
    result: Mapping[str, Any],
    *,
    report_output_path: object,
) -> str:
    histories = _mapping(result.get("highlight_run_histories"))
    rows: list[str] = []
    for role in ("best", "worst"):
        entry = _mapping(histories.get(role))
        if not entry:
            continue
        links = []
        for label, key in (("Timeline", "html_path"), ("JSON", "json_path"), ("Map", "map_path")):
            path = entry.get(key)
            if path:
                href = _relative_link(report_output_path, path)
                links.append(f'<a href="{html_escape(href)}">{label}</a>')
        link_html = " / ".join(links) or "-"
        rows.append(
            "<tr>"
            f"<td>{html_escape(role.title())}</td>"
            f"<td>{html_escape(str(entry.get('run_index', '')))}</td>"
            f"<td>{html_escape(str(entry.get('generated_at', '')))}</td>"
            f"<td>{html_escape(str(entry.get('seed', '')))}</td>"
            f"<td>{_float(entry.get('total_reward')):.3f}</td>"
            f"<td>{_int(entry.get('final_act'))}</td>"
            f"<td>{_int(entry.get('final_floor'))}</td>"
            f"<td>{html_escape(str(entry.get('final_phase', '')))}</td>"
            f"<td>{link_html}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<h2>Best And Worst Evaluation Run Histories</h2>"
        '<div class="scroll"><table>'
        "<thead><tr><th>Role</th><th>Run</th><th>Generated</th><th>Seed</th><th>Reward</th>"
        "<th>Act</th><th>Floor</th><th>Phase</th><th>Artifacts</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _relative_link(report_output_path: object, target_path: object) -> str:
    target = Path(str(target_path))
    if report_output_path in {None, ""}:
        return target.as_posix()
    report = Path(str(report_output_path))
    try:
        return os.path.relpath(target, start=report.parent).replace("\\", "/")
    except ValueError:
        return target.as_posix()


def _optional_path(value: object) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value))


def _write_ppo_progress_report(
    result: Mapping[str, Any],
    path: Path | str,
    *,
    progress: Sequence[LearningProgressPoint],
    window: int,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _ppo_progress_html(result, progress=progress, window=window),
        encoding="utf-8",
    )


def _ppo_progress_html(
    result: Mapping[str, Any],
    *,
    progress: Sequence[LearningProgressPoint],
    window: int,
) -> str:
    normalized = with_moving_averages(progress, window=window)
    metadata = _mapping(result.get("metadata"))
    batches = tuple(
        batch
        for batch in _sequence(result.get("batch_summaries"))
        if isinstance(batch, Mapping)
    )
    last_batch = _mapping(batches[-1]) if batches else {}
    planning_rows = _planning_rows_html(batches[-12:])
    reward_rows = _reward_component_rows_html(batches[-12:])
    diagnostic_rows = _diagnostic_rows_html(batches[-12:])
    highlight_links = _highlight_links_html(
        result,
        report_output_path=result.get("report_output_path"),
    )
    summary = {
        "runs_trained": _int(result.get("runs_trained")),
        "total_steps": _int(result.get("total_steps")),
        "batches_completed": _int(result.get("batches_completed")),
        "average_training_reward": _float(result.get("average_training_reward")),
        "latest_eval_reward": _float(last_batch.get("evaluation_average_reward")),
        "latest_eval_floor": _float(last_batch.get("evaluation_average_floor")),
        "latest_success_rate": _float(last_batch.get("evaluation_target_success_rate")),
        "parameter_count": _int(metadata.get("parameter_count")),
    }
    summary_items = "\n".join(
        f"<div><strong>{html_escape(key.replace('_', ' ').title())}</strong>"
        f"<span>{html_escape(str(value))}</span></div>"
        for key, value in summary.items()
    )
    latest_progress_rows = "\n".join(
        "<tr>"
        f"<td>{point.run_index}</td>"
        f"<td>{html_escape(str(point.seed))}</td>"
        f"<td>{point.steps_taken}</td>"
        f"<td>{point.total_reward:.3f}</td>"
        f"<td>{point.final_act}</td>"
        f"<td>{point.final_floor}</td>"
        f"<td>{html_escape(point.final_phase)}</td>"
        "</tr>"
        for point in normalized[-20:]
    )
    schema_text = (
        f"Schema v{html_escape(str(metadata.get('network_schema_version', '')))} - "
        f"{html_escape(str(metadata.get('character_id', '')))} "
        f"A{html_escape(str(metadata.get('ascension', '')))} - "
        "planning heads included."
    )
    planning_headers = _planning_table_headers()
    reward_headers = _reward_component_table_headers(batches)
    diagnostic_headers = _diagnostic_table_headers(batches)
    latest_headers = (
        "<th>Run</th><th>Seed</th><th>Steps</th><th>Reward</th>"
        "<th>Act</th><th>Floor</th><th>Phase</th>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Masked PPO Training Progress</title>
  <style>
    body {{
      margin: 0;
      background: #f5f7f8;
      color: #172026;
      font: 14px/1.45 system-ui, sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    h1 {{ margin: 0 0 4px; font-size: 28px; }}
    h2 {{ margin: 28px 0 10px; font-size: 18px; }}
    .muted {{ color: #65717a; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .summary div {{
      background: #fff;
      border: 1px solid #d9e0e5;
      border-radius: 8px;
      padding: 12px;
    }}
    .summary strong {{
      display: block;
      color: #57636c;
      font-size: 12px;
      text-transform: uppercase;
    }}
    .summary span {{
      display: block;
      margin-top: 5px;
      font-size: 19px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid #d9e0e5;
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e6ebef;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #eef2f5;
      color: #4d5961;
      font-size: 12px;
      text-transform: uppercase;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .scroll {{ overflow-x: auto; }}
  </style>
</head>
<body>
<main>
  <h1>Masked PPO Training Progress</h1>
  <p class="muted">{schema_text}</p>
  <section class="summary">{summary_items}</section>
  {highlight_links}
  <h2>Planning Head Trends</h2>
  <div class="scroll">
    <table>
      <thead>
        <tr><th>Batch</th><th>Eval Reward</th><th>Eval Floor</th>{planning_headers}</tr>
      </thead>
      <tbody>{planning_rows}</tbody>
    </table>
  </div>
	  <h2>Reward Component Trends</h2>
	  <div class="scroll">
	    <table>
	      <thead>
	        <tr><th>Batch</th>{reward_headers}</tr>
      </thead>
      <tbody>{reward_rows}</tbody>
	    </table>
	  </div>
	  <h2>Reward And Deck Diagnostics</h2>
	  <div class="scroll">
	    <table>
	      <thead>
	        <tr><th>Batch</th>{diagnostic_headers}</tr>
	      </thead>
	      <tbody>{diagnostic_rows}</tbody>
	    </table>
	  </div>
	  <h2>Latest Training Runs</h2>
  <div class="scroll">
    <table>
      <thead><tr>{latest_headers}</tr></thead>
      <tbody>{latest_progress_rows}</tbody>
    </table>
  </div>
</main>
</body>
</html>
"""


def _planning_table_headers() -> str:
    return "".join(
        f"<th>{html_escape(key.replace('_', ' ').title())}</th>"
        for key in PLANNING_HEAD_SCHEMA
    )


def _planning_rows_html(batches: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for batch in batches:
        averages = _mapping(batch.get("planning_output_averages"))
        plan_cells = "".join(
            f"<td>{_float(averages.get(key)):.3f}</td>" for key in PLANNING_HEAD_SCHEMA
        )
        rows.append(
            "<tr>"
            f"<td>{_int(batch.get('batch_index'))}</td>"
            f"<td>{_float(batch.get('evaluation_average_reward')):.3f}</td>"
            f"<td>{_float(batch.get('evaluation_average_floor')):.3f}</td>"
            f"{plan_cells}"
            "</tr>"
        )
    return "\n".join(rows)


def _reward_component_keys(batches: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    preferred = (
        "total",
        "aggression_pressure",
        "hp_loss_penalty",
        "enemy_hp_progress_reward",
        "prevented_hp_reward",
        "combat_win_reward",
        "boss_reward",
        "combat_pace_reward",
        "node_progress_reward",
        "gold_reward",
        "potion_waste_penalty",
        "resource_pickup_reward",
        "reward_skip_penalty",
        "deck_capability_reward",
        "deck_burden_penalty",
        "starter_deck_similarity_penalty",
        "target_reached_reward",
    )
    present = {
        str(key)
        for batch in batches
        for key in _mapping(batch.get("reward_component_averages"))
    }
    ordered = [key for key in preferred if key in present]
    ordered.extend(sorted(present - set(ordered)))
    return tuple(ordered)


def _reward_component_table_headers(batches: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        f"<th>{html_escape(key.replace('_', ' ').title())}</th>"
        for key in _reward_component_keys(batches)
    )


def _reward_component_rows_html(batches: Sequence[Mapping[str, Any]]) -> str:
    keys = _reward_component_keys(batches)
    rows: list[str] = []
    for batch in batches:
        averages = _mapping(batch.get("reward_component_averages"))
        cells = "".join(f"<td>{_float(averages.get(key)):.3f}</td>" for key in keys)
        rows.append(
            "<tr>"
            f"<td>{_int(batch.get('batch_index'))}</td>"
            f"{cells}"
            "</tr>"
        )
    return "\n".join(rows)


def _diagnostic_keys(batches: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    preferred = (
        "take_reward_card",
        "take_reward_gold",
        "take_reward_potion",
        "take_reward_relic",
        "skip_reward_card_options",
        "skip_reward_card_group",
        "skip_reward_fixed_card",
        "skip_reward_gold",
        "skip_reward_potion",
        "skip_reward_relic",
        "proceed_with_unclaimed_card",
        "proceed_with_unclaimed_card_removal",
        "proceed_with_unclaimed_gold",
        "proceed_with_unclaimed_potion",
        "proceed_with_unclaimed_relic",
        "deck_cards_added",
        "deck_cards_removed",
        "relics_gained",
        "final_deck_size",
        "final_unknown_card_count",
        "final_relic_count",
        "final_potion_count",
        "final_gold",
    )
    present = {
        str(key)
        for batch in batches
        for key in _mapping(batch.get("diagnostic_averages"))
    }
    ordered = [key for key in preferred if key in present]
    ordered.extend(sorted(present - set(ordered)))
    return tuple(ordered)


def _diagnostic_table_headers(batches: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        f"<th>{html_escape(key.replace('_', ' ').title())}</th>"
        for key in _diagnostic_keys(batches)
    )


def _diagnostic_rows_html(batches: Sequence[Mapping[str, Any]]) -> str:
    keys = _diagnostic_keys(batches)
    rows: list[str] = []
    for batch in batches:
        averages = _mapping(batch.get("diagnostic_averages"))
        cells = "".join(f"<td>{_float(averages.get(key)):.3f}</td>" for key in keys)
        rows.append(
            "<tr>"
            f"<td>{_int(batch.get('batch_index'))}</td>"
            f"{cells}"
            "</tr>"
        )
    return "\n".join(rows)


def _load_checkpoint_if_available(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    device: Any,
    expected_architecture: Mapping[str, Any],
    resume: bool,
    resume_from_path: Path | str | None,
    model_output_path: Path | str | None,
) -> _ResumeState:
    if not resume:
        return _ResumeState()
    for candidate in (
        Path(resume_from_path) if resume_from_path is not None else None,
        Path(model_output_path) if model_output_path is not None else None,
    ):
        if candidate is None or not candidate.exists():
            continue
        payload = torch.load(candidate, map_location=device)
        if isinstance(payload, Mapping):
            architecture = payload.get("architecture")
            if isinstance(architecture, Mapping) and not _architecture_matches(
                architecture,
                expected_architecture,
            ):
                raise RuntimeError(
                    "Refusing to resume from an incompatible PPO checkpoint. "
                    f"Checkpoint architecture={dict(architecture)!r}; "
                    f"expected={dict(expected_architecture)!r}. "
                    "Start a fresh run or choose a checkpoint with matching "
                    "network_schema_version and dimensions."
                )
            if not isinstance(architecture, Mapping):
                raise RuntimeError(
                    "Refusing to resume from a PPO checkpoint without architecture "
                    "metadata. Start a fresh run with resume disabled."
                )
            model_state = payload.get("model_state")
            optimizer_state = payload.get("optimizer_state")
            if model_state is not None:
                try:
                    model.load_state_dict(model_state)
                except RuntimeError:
                    continue
            if optimizer_state is not None:
                try:
                    optimizer.load_state_dict(optimizer_state)
                    _move_optimizer_state_to_device(optimizer, device)
                except ValueError:
                    optimizer_state = None
            result = payload.get("result")
            return _ResumeState(
                path=str(candidate),
                result=result if isinstance(result, Mapping) else None,
            )
    return _ResumeState()


def _move_optimizer_state_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            to_device = getattr(value, "to", None)
            if callable(to_device):
                state[key] = to_device(device)


def _resume_progress_points(
    previous_result: Mapping[str, Any],
    key: str,
) -> tuple[LearningProgressPoint, ...]:
    points: list[LearningProgressPoint] = []
    for item in _sequence(previous_result.get(key)):
        if not isinstance(item, Mapping):
            continue
        try:
            points.append(LearningProgressPoint.model_validate(item))
        except ValueError:
            continue
    return tuple(points)


def _resume_total_steps(
    previous_result: Mapping[str, Any],
    training_points: Sequence[LearningProgressPoint],
) -> int:
    total_steps = _int(previous_result.get("total_steps"))
    if total_steps > 0:
        return total_steps
    return sum(point.steps_taken for point in training_points)


def _resume_total_reward(
    previous_result: Mapping[str, Any],
    training_points: Sequence[LearningProgressPoint],
) -> float:
    if training_points:
        return sum(point.total_reward for point in training_points)
    return _float(previous_result.get("average_training_reward")) * max(
        0,
        _int(previous_result.get("runs_trained")),
    )


def _advance_run_seed_rng(rng: random.Random, used_seed_count: int) -> None:
    for _index in range(max(0, used_seed_count)):
        _random_run_seed(rng)


def _checkpoint_architecture(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(result.get("metadata"))
    return {
        "network_schema_version": _int(metadata.get("network_schema_version")),
        "observation_dim": _int(metadata.get("observation_dim")),
        "action_feature_dim": _int(metadata.get("action_feature_dim")),
        "content_vocab_schema_version": _int(metadata.get("content_vocab_schema_version")),
        "content_vocab_size": _int(metadata.get("content_vocab_size")),
        "content_vocab_checksum": str(metadata.get("content_vocab_checksum", "")),
        "content_identity_slots": _int(metadata.get("content_identity_slots")),
        "content_identity_embedding_dim": _int(
            metadata.get("content_identity_embedding_dim")
        ),
        "planning_head_dim": _int(metadata.get("planning_head_dim")),
        "planning_head_schema": list(_sequence(metadata.get("planning_head_schema"))),
        "hidden_size": _int(metadata.get("hidden_size")),
        "hidden_layers": _int(metadata.get("hidden_layers")),
        "head_hidden_layers": _int(metadata.get("head_hidden_layers")),
        "activation": str(metadata.get("activation", "")),
    }


def _architecture_matches(
    checkpoint_architecture: Mapping[str, Any],
    expected_architecture: Mapping[str, Any],
) -> bool:
    keys = (
        "network_schema_version",
        "observation_dim",
        "action_feature_dim",
        "content_vocab_schema_version",
        "content_vocab_size",
        "content_vocab_checksum",
        "content_identity_slots",
        "content_identity_embedding_dim",
        "planning_head_dim",
        "planning_head_schema",
        "hidden_size",
        "hidden_layers",
        "head_hidden_layers",
        "activation",
    )
    return all(
        str(checkpoint_architecture.get(key)) == str(expected_architecture.get(key))
        for key in keys
    )


def _load_torch() -> tuple[Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        optim = importlib.import_module("torch.optim")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Masked PPO requires PyTorch. Install the RL extras with "
            "`uv sync --extra rl`, then rerun the command."
        ) from exc
    return torch, nn, optim


def _resolve_torch_device(torch: Any, requested: str) -> Any:
    normalized = str(requested or "auto").strip().lower()
    if normalized in {"auto", ""}:
        return torch.device("cuda" if bool(torch.cuda.is_available()) else "cpu")
    if normalized in {"gpu", "cuda"}:
        if not bool(torch.cuda.is_available()):
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot see a CUDA device. "
                "Install a CUDA-enabled PyTorch build, then verify with "
                "`uv run python -c \"import torch; print(torch.cuda.is_available())\"`."
            )
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    try:
        return torch.device(normalized)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            "device must be 'auto', 'cpu', 'cuda', or a valid torch device string "
            "such as 'cuda:0'."
        ) from exc


def _torch_device_metadata(
    torch: Any,
    device: Any,
    *,
    requested_device: str,
) -> dict[str, Any]:
    device_text = str(device)
    is_cuda = device_text.startswith("cuda")
    device_index = getattr(device, "index", None)
    if is_cuda and device_index is None:
        device_index = torch.cuda.current_device()
    return {
        "requested_device": str(requested_device),
        "device": device_text,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": (
            str(torch.cuda.get_device_name(device_index)) if is_cuda else None
        ),
    }


def _character_display_name(character_id: object) -> str:
    normalized = str(character_id or "").strip().upper()
    return {
        "IRONCLAD": "The Ironclad",
        "SILENT": "The Silent",
        "DEFECT": "The Defect",
        "WATCHER": "The Watcher",
        "NECROBINDER": "The Necrobinder",
    }.get(normalized, normalized or "Unknown")


def _action_space(info: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = info.get("action_space")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return tuple(item for item in raw if isinstance(item, Mapping))
    return ()


def _lookup_vector_int(observation: Mapping[str, Any], field: str) -> int:
    schema = observation.get("vector_schema")
    vector = observation.get("vector")
    if isinstance(schema, list | tuple) and isinstance(vector, list | tuple):
        for index, name in enumerate(schema):
            if str(name) == field and index < len(vector):
                return _int(vector[index])
    return 0


def _random_run_seed(rng: random.Random) -> int:
    return rng.randrange(0, 2_147_483_647)


def _write_json(value: Mapping[str, Any], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_bucket(value: str, bucket_count: int) -> int:
    if bucket_count <= 1:
        return 0
    total = 0
    for character in value:
        total = ((total * 33) + ord(character)) % bucket_count
    return total


def _scaled(value: int | float, maximum: int | float) -> float:
    return max(0.0, min(1.0, float(value) / max(1.0, float(maximum))))


def _scaled_fraction(value: object) -> float:
    return max(0.0, min(1.0, _float(value)))


def _signed_scaled(value: float, maximum: float) -> float:
    limit = max(1.0, abs(float(maximum)))
    return max(-1.0, min(1.0, float(value) / limit))


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _float(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _int(value: object) -> int:
    return int(_float(value))


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _normalized_id(value: object) -> str:
    text = str(value or "").strip().lower()
    normalized = []
    previous_sep = False
    for character in text:
        if character.isalnum():
            normalized.append(character)
            previous_sep = False
        elif not previous_sep:
            normalized.append("_")
            previous_sep = True
    return "".join(normalized).strip("_")

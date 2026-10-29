"""Validated, reusable inference for saved masked PPO policies."""

from __future__ import annotations

import math
import operator
import pickle
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2sim.agent_api import action_mask, action_space, encode_observation
from sts2sim.learning import masked_ppo as ppo


class CheckpointPolicyError(ValueError):
    """A checkpoint cannot safely provide a compatible trained policy."""


class PolicyInferenceError(ValueError):
    """An observation or legal-action contract cannot be used for inference."""


@dataclass(frozen=True)
class MaskedPpoDecision:
    action_id: int
    action: dict[str, Any]
    descriptor: dict[str, Any]
    policy_output: dict[str, float]


def _validated_checkpoint(
    checkpoint_path: Path | str,
) -> tuple[Any, Any, Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path)
    try:
        torch, nn, _ = ppo._load_torch()
        payload = ppo._read_ppo_checkpoint(torch, path, "cpu")
        architecture = payload.get("architecture")
        if not isinstance(architecture, Mapping):
            raise ValueError("checkpoint has no architecture metadata")
        configuration: dict[str, Any] = {}
        for key in ("hidden_size", "hidden_layers", "head_hidden_layers"):
            value = architecture.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"checkpoint architecture {key} must be a positive integer")
            configuration[key] = value
        activation = architecture.get("activation")
        if not isinstance(activation, str):
            raise ValueError("checkpoint architecture activation is missing")
        configuration["activation"] = ppo._normalize_activation_name(activation)
        compatibility = ppo.validate_masked_ppo_checkpoint(path, **configuration)
        if not compatibility.get("compatible"):
            keys = ", ".join(str(item["key"]) for item in compatibility.get("mismatches", ()))
            reason = str(compatibility.get("reason", "incompatible checkpoint"))
            raise ValueError(f"{reason}{': ' + keys if keys else ''}")
        # Recheck this payload as well as the public validator's file read. A
        # checkpoint replaced during validation must not supply stale metadata.
        expected = ppo._expected_ppo_architecture(**configuration)
        if ppo._architecture_mismatches(architecture, expected):
            raise ValueError("checkpoint changed during validation or has incompatible metadata")
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointPolicyError(f"Cannot use PPO checkpoint {path}: {exc}") from exc
    return torch, nn, payload, configuration, compatibility


class MaskedPpoPolicy:
    """Load compatible saved weights and choose exclusively from allowed actions.

    ``predict`` consumes the observation/info pair returned by ``Sts2Env``.
    ``choose_action`` builds the same inputs directly from a simulator state.
    Architecture dimensions, content vocabulary, and simulator contracts are
    checked by the trainer's public checkpoint validator. Loading never creates
    a substitute random policy or enables unrestricted pickle deserialization.
    """

    def __init__(self, checkpoint_path: Path | str, *, device: str = "auto") -> None:
        torch, nn, payload, configuration, compatibility = _validated_checkpoint(checkpoint_path)
        self.checkpoint_path = str(Path(checkpoint_path))
        self.architecture = dict(payload["architecture"])
        self.configuration = configuration
        self.compatibility = compatibility
        self._torch = torch
        try:
            self.device = ppo._resolve_torch_device(torch, device)
            with torch.random.fork_rng(devices=[]):
                self.model = ppo._masked_actor_critic_class(nn)(
                    **ppo._model_kwargs_from_architecture(self.architecture)
                )
            ppo._load_model_state(torch, self.model, payload.get("model_state"))
            self.model.to(self.device)
            self.model.eval()
        except (RuntimeError, ValueError, TypeError) as exc:
            raise CheckpointPolicyError(f"Cannot initialize PPO inference: {exc}") from exc

    def predict(
        self,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
        *,
        deterministic: bool = True,
        allowed_action_ids: Sequence[int] | None = None,
    ) -> MaskedPpoDecision:
        """Score masked legal descriptors and return an action plus diagnostics."""
        try:
            vector = tuple(float(value) for value in observation["vector"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyInferenceError("Observation must contain a numeric vector.") from exc
        if len(vector) != self.architecture["observation_dim"] or not all(
            math.isfinite(value) for value in vector
        ):
            raise PolicyInferenceError(
                "Observation vector has an incompatible shape or non-finite values."
            )
        raw_mask = observation.get("action_mask", info.get("action_mask"))
        try:
            mask = list(raw_mask)
        except TypeError as exc:
            raise PolicyInferenceError("Inference requires an explicit legal-action mask.") from exc
        if any(value not in (0, 1) for value in mask):
            raise PolicyInferenceError("The legal-action mask must be binary.")
        if "action_mask" in observation and "action_mask" in info:
            try:
                masks_agree = mask == list(info["action_mask"])
            except TypeError as exc:
                raise PolicyInferenceError("Info has an invalid action mask.") from exc
            if not masks_agree:
                raise PolicyInferenceError("Observation and info action masks disagree.")
        allowed: set[int] | None = None
        if allowed_action_ids is not None:
            try:
                if any(isinstance(value, bool) for value in allowed_action_ids):
                    raise TypeError
                allowed = {operator.index(value) for value in allowed_action_ids}
            except TypeError as exc:
                raise PolicyInferenceError("Allowed action IDs must be integers.") from exc
        descriptors: list[Mapping[str, Any]] = []
        seen: set[int] = set()
        for descriptor in ppo._action_space(info):
            try:
                raw_id = descriptor["id"]
                if isinstance(raw_id, bool):
                    raise TypeError
                action_id = operator.index(raw_id)
            except (KeyError, TypeError) as exc:
                raise PolicyInferenceError("Action descriptors require integer IDs.") from exc
            if action_id < 0 or action_id >= len(mask) or action_id in seen:
                raise PolicyInferenceError(
                    "Action descriptor IDs are duplicated or outside the mask."
                )
            seen.add(action_id)
            if mask[action_id] and (allowed is None or action_id in allowed):
                if not isinstance(descriptor.get("action"), Mapping):
                    raise PolicyInferenceError("A legal descriptor has no engine action payload.")
                descriptors.append(descriptor)
        if not descriptors:
            raise PolicyInferenceError("No supported legal actions remain in the policy mask.")
        filtered_info = {"action_space": descriptors}
        try:
            result = ppo._choose_action(
                self._torch,
                self.model,
                observation,
                filtered_info,
                deterministic=deterministic,
                device=self.device,
            )
        except (ValueError, RuntimeError, TypeError, IndexError) as exc:
            raise PolicyInferenceError(f"PPO inference failed: {exc}") from exc
        if result is None:
            raise PolicyInferenceError("The policy returned no legal action.")
        action_id, index, _, _, _, _, context = result
        if not all(math.isfinite(value) for value in context.values()):
            raise PolicyInferenceError("The policy returned non-finite decision diagnostics.")
        descriptor = descriptors[index]
        return MaskedPpoDecision(
            action_id=action_id,
            action=dict(descriptor["action"]),
            descriptor=dict(descriptor),
            policy_output=dict(context),
        )

    def choose_action_id(
        self,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
        *,
        deterministic: bool = True,
    ) -> int:
        return self.predict(observation, info, deterministic=deterministic).action_id

    def choose_action(
        self,
        state: Any,
        *,
        agent_memory: object | None = None,
        allowed_action_ids: Sequence[int] | None = None,
    ) -> MaskedPpoDecision:
        descriptors = action_space(state)
        observation = encode_observation(state, include_state=False, agent_memory=agent_memory)
        observation["action_mask"] = list(action_mask(state))
        return self.predict(
            observation,
            {"action_space": descriptors},
            allowed_action_ids=allowed_action_ids,
        )


def evaluate_ppo(
    checkpoint_path: Path | str,
    *,
    seed: int | str,
    target: str = "act1-boss",
    eval_runs: int = 20,
    eval_max_steps: int = 1200,
    character_id: str = "IRONCLAD",
    ascension: int = 0,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate saved weights on a reproducible seed suite without training."""
    if eval_runs < 1 or eval_max_steps < 1:
        raise ValueError("Evaluation requires at least one run and one step.")
    _, _, _, configuration, _ = _validated_checkpoint(checkpoint_path)
    resolved_target = ppo.resolve_ppo_target(target)
    result = ppo.evaluate_masked_ppo_checkpoint(
        checkpoint_path,
        target=resolved_target,
        seed=seed,
        eval_runs=eval_runs,
        eval_max_steps=eval_max_steps,
        character_id=character_id,
        ascension=ascension,
        device=device,
        **configuration,
    )
    suite_rng = random.Random(f"{seed}:checkpoint:{resolved_target.name}")
    return {
        **result,
        "seed": seed,
        "character_id": character_id,
        "ascension": ascension,
        "episode_seeds": [ppo._random_run_seed(suite_rng) for _ in range(eval_runs)],
        "evaluation_max_steps": eval_max_steps,
        "architecture_configuration": configuration,
    }


__all__ = [
    "CheckpointPolicyError",
    "MaskedPpoDecision",
    "MaskedPpoPolicy",
    "PolicyInferenceError",
    "evaluate_ppo",
]

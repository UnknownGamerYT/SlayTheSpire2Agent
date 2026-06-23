"""Combat action scoring for the strategic agent skeleton."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .evaluators import action_payload, action_type, make_score, mapping, normalized, number
from .models import ActionDescriptor, DecisionContext, ScoredAction


class CombatPlanner:
    """Heuristic combat planner over current legal combat actions."""

    def score(self, context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
        kind = action_type(descriptor)
        if kind == "play_card":
            return self._score_play_card(context, descriptor)
        if kind == "end_turn":
            return self._score_end_turn(context, descriptor)
        if kind == "use_potion":
            return self._score_use_potion(context, descriptor)
        if kind == "discard_potion":
            return make_score(
                descriptor,
                score=-25.0,
                category="combat",
                reasons=("discarding_potion_is_only_for_slot_management",),
            )
        if kind in {"choose_card", "discard_card", "exhaust_card"}:
            return self._score_pending_card_choice(context, descriptor)
        return make_score(
            descriptor,
            score=0.0,
            category="combat",
            reasons=("combat_legal_action_without_special_case",),
        )

    def _score_play_card(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        payload = action_payload(descriptor)
        combat = mapping(context.state_summary.get("combat"))
        card = _card_by_instance_id(_sequence(combat.get("hand")), payload.get("card_instance_id"))
        card_type = normalized(card.get("type"))
        card_id = normalized(card.get("card_id"))
        effects = mapping(card.get("effects"))
        preview = mapping(descriptor.get("preview"))
        cost = number(card.get("cost"), 1.0)
        damage = _effect_amount(effects, "damage")
        block = _effect_amount(effects, "block")
        draw = _effect_amount(effects, "draw")
        energy = _effect_amount(effects, "energy")
        reasons: list[str] = []
        score = 10.0

        if card_type == "attack":
            pace_bonus = 1.45 if context.plan.aggression.combat_pace == "rush" else 1.2
            score += 8.0 + max(damage, 6.0) * pace_bonus
            reasons.append("attack_advances_lethal_plan")
            if context.plan.aggression.scaling_pressure >= 0.45:
                score += damage * 0.45
                reasons.append("scaling_enemy_rewards_faster_kill")
        elif card_type == "skill":
            score += 5.0 + block * (0.7 + context.plan.aggression.block_priority) + draw * 2.0
            reasons.append("skill_improves_turn_state")
        elif card_type == "power":
            score += 18.0
            reasons.append("power_adds_scaling")
        else:
            score += 3.0
            reasons.append("card_is_playable")

        if block > 0:
            incoming = context.plan.threat.incoming_damage
            projected_damage = number(preview.get("projected_damage_taken_after_end"), incoming)
            if projected_damage > context.plan.aggression.hp_spend_budget:
                score += min(block, incoming) * (1.4 + context.plan.aggression.block_priority)
                reasons.append("block_protects_hp_budget")
            else:
                score += min(block, incoming) * 0.55
                reasons.append("hp_budget_allows_some_chip_damage")
        if draw > 0:
            score += draw * 2.5
            reasons.append("draw_improves_options")
        if energy > 0:
            score += energy * 2.0
            reasons.append("energy_generation_extends_turn")
        if _target_would_die(combat, payload.get("target_id"), damage):
            score += 12.0
            reasons.append("likely_kill_target")
        if cost > 0:
            score -= cost * 1.5
        if card_id in {normalized(item) for item in context.plan.upgrade_targets}:
            score += 1.0
            reasons.append("plan_already_values_this_card")

        return make_score(
            descriptor,
            score=score,
            category="combat",
            reasons=tuple(reasons) or ("play_card_progress",),
        )

    def _score_end_turn(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        playable_count = sum(
            1 for action in context.legal_actions if action_type(action) == "play_card"
        )
        preview = mapping(descriptor.get("preview"))
        projected_damage = number(
            preview.get("projected_damage_taken_after_end"),
            context.plan.threat.incoming_damage,
        )
        score = 2.0
        reasons = ["end_turn_is_always_legal"]
        if playable_count:
            score -= 12.0
            reasons.append("playable_cards_remain")
        if context.plan.threat.incoming_damage > 0 and context.plan.risk_level in {
            "high",
            "critical",
        }:
            score -= 8.0
            reasons.append("incoming_damage_requires_care")
        if projected_damage > context.plan.aggression.hp_spend_budget:
            score -= min(18.0, (projected_damage - context.plan.aggression.hp_spend_budget) * 1.2)
            reasons.append("projected_damage_exceeds_hp_spend_budget")
        elif projected_damage > 0 and context.plan.aggression.allow_chip_damage:
            score += 2.0
            reasons.append("chip_damage_inside_aggression_budget")
        if (
            playable_count
            and context.plan.aggression.combat_pace == "rush"
            and context.plan.aggression.scaling_pressure >= 0.45
        ):
            score -= 8.0
            reasons.append("rushing_scaling_enemy_discourages_early_end_turn")
        return make_score(
            descriptor,
            score=score,
            category="combat",
            reasons=tuple(reasons),
        )

    def _score_use_potion(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        payload = action_payload(descriptor)
        potion_id = normalized(mapping(payload.get("payload")).get("potion_id"))
        potion_strategy = mapping(descriptor.get("potion_strategy"))
        score = 4.0
        reasons = ["potion_can_convert_to_combat_advantage"]
        if context.plan.risk_level in {"high", "critical"}:
            score += 14.0
            reasons.append("high_risk_state_allows_potion_spend")
        if potion_strategy.get("lethal_now") or potion_strategy.get("survival_enabling"):
            score += 18.0
            reasons.append("potion_solves_current_fight_tactical_need")
        elif context.plan.aggression.combat_pace == "rush" and potion_strategy.get(
            "preemptive_fight_setup"
        ):
            score += 8.0
            reasons.append("rush_plan_allows_setup_potion")
        elif context.plan.aggression.target < 0.35:
            score -= 5.0
            reasons.append("safe_plan_saves_noncritical_potion")
        if context.plan.threat.phase == "combat" and context.plan.threat.alive_monsters:
            score += 2.0
        if "fairy" in potion_id:
            score -= 100.0
            reasons.append("automatic_save_potion_should_not_be_spent")
        return make_score(
            descriptor,
            score=score,
            category="combat",
            reasons=tuple(reasons),
        )

    def _score_pending_card_choice(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        payload = action_payload(descriptor)
        combat = mapping(context.state_summary.get("combat"))
        card = _card_by_instance_id(_all_combat_cards(combat), payload.get("card_instance_id"))
        card_id = normalized(card.get("card_id"))
        kind = action_type(descriptor)
        score = 8.0
        reasons = [f"{kind}_resolves_pending_choice"]
        if kind in {"discard_card", "exhaust_card"}:
            if (
                "strike" in card_id
                or "defend" in card_id
                or normalized(card.get("type")) == "curse"
            ):
                score += 6.0
                reasons.append("low_value_card_selected")
            else:
                score -= 2.0
                reasons.append("preserve_stronger_cards_when_possible")
        return make_score(
            descriptor,
            score=score,
            category="combat",
            reasons=tuple(reasons),
        )


def _effect_amount(effects: Mapping[str, Any], key: str) -> float:
    value = effects.get(key)
    if isinstance(value, Mapping):
        return number(value.get("amount"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return float(len(value))
    return number(value)


def _card_by_instance_id(cards: Sequence[object], instance_id: object) -> dict[str, Any]:
    wanted = str(instance_id or "")
    for raw_card in cards:
        card = mapping(raw_card)
        if str(card.get("instance_id", "")) == wanted:
            return card
    return {}


def _all_combat_cards(combat: Mapping[str, Any]) -> tuple[object, ...]:
    cards: list[object] = []
    for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        cards.extend(_sequence(combat.get(zone)))
    return tuple(cards)


def _target_would_die(combat: Mapping[str, Any], target_id: object, damage: float) -> bool:
    if damage <= 0:
        return False
    wanted = str(target_id or "")
    for raw_monster in _sequence(combat.get("monsters")):
        monster = mapping(raw_monster)
        if wanted and str(monster.get("monster_id", "")) != wanted:
            continue
        hp = number(monster.get("hp"))
        block = number(monster.get("block"))
        if hp > 0 and max(0.0, damage - block) >= hp:
            return True
    return False


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()

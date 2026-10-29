from combat.test_card_play_status_runtime import _simple

from sts2sim.engine import CardEnchantment
from sts2sim.engine.source_enchantment_hooks import can_enchant, refresh_properties


def test_source_enchantment_type_and_single_enchantment_constraints():
    for kind in ("curse", "status", "quest"):
        assert not can_enchant(_simple("bad", card_type=kind), "swift")
    for name in ("sharp", "vigorous", "corrupted", "instinct", "momentum"):
        assert can_enchant(_simple("attack", card_type="attack"), name)
        assert not can_enchant(_simple("skill"), name)
    enchanted = _simple("occupied", enchantments=(CardEnchantment(keyword="steady", amount=1),))
    assert not can_enchant(enchanted, "swift")
    locked = _simple("locked", custom={"unplayable": True})
    assert not can_enchant(locked, "swift")
    assert can_enchant(locked, "swift", in_deck=False)
    assert not can_enchant(locked, "slither", in_deck=False)


def test_source_enchantment_special_eligibility_uses_tags_and_native_properties():
    defend = _simple("defend_ironclad")
    strike = _simple("strike_ironclad", card_type="attack")
    assert can_enchant(defend, "goopy")
    assert can_enchant(defend, "spiral")
    assert can_enchant(strike, "spiral")
    assert not can_enchant(_simple("fake_defend_name"), "goopy")
    assert not can_enchant(_simple("other"), "spiral")
    assert can_enchant(_simple("fisticuffs", card_type="attack"), "nimble")
    assert not can_enchant(_simple("inflame", card_type="power"), "nimble")
    assert can_enchant(_simple("shiv", card_type="attack", exhausts=True), "souls_power")
    assert not can_enchant(_simple("permanent"), "souls_power")
    assert not can_enchant(_simple("x", cost=-1, custom={"x_cost": True}), "slither")


def test_source_on_enchant_properties_apply_without_changing_native_effects():
    cases = {
        "royally_approved": {"innate": True, "retain": True},
        "steady": {"retain": True},
        "tezcataras_ember": {"eternal": True},
    }
    for name, expected in cases.items():
        card = _simple(
            "card",
            cost=3,
            effects={"block": 5},
            enchantments=(CardEnchantment(keyword=name, amount=1),),
        )
        card = refresh_properties(card)
        assert all(card.custom.get(k) == v for k, v in expected.items())
        assert card.effects == {"block": 5}
        assert card.cost == (0 if name == "tezcataras_ember" else 3)
    goopy = refresh_properties(
        _simple("defend_ironclad", enchantments=(CardEnchantment(keyword="goopy", amount=1),))
    )
    assert goopy.exhausts
    souls = refresh_properties(
        _simple(
            "shiv",
            card_type="attack",
            exhausts=True,
            enchantments=(CardEnchantment(keyword="souls_power", amount=1),),
        )
    )
    assert not souls.exhausts

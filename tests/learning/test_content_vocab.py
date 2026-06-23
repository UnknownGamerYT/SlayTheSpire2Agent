from __future__ import annotations

from sts2sim.learning.content_vocab import (
    CONTENT_IDENTITY_SLOTS,
    descriptor_identity_ids,
    descriptor_identity_tokens,
    load_content_vocab,
)


def test_content_vocab_loads_exact_cached_content_ids() -> None:
    vocab = load_content_vocab()

    anchor = vocab.id_for_token("relic:anchor")
    membership_card = vocab.id_for_token("relic:membership_card")
    pommel_strike = vocab.id_for_token("card:pommel_strike")
    shrug_it_off = vocab.id_for_token("card:shrug_it_off")

    assert vocab.size > 1000
    assert anchor not in {vocab.pad_id, vocab.unknown_id}
    assert membership_card not in {vocab.pad_id, vocab.unknown_id}
    assert pommel_strike not in {vocab.pad_id, vocab.unknown_id}
    assert shrug_it_off not in {vocab.pad_id, vocab.unknown_id}
    assert len({anchor, membership_card, pommel_strike, shrug_it_off}) == 4


def test_descriptor_identity_ids_distinguish_same_shape_reward_choices() -> None:
    vocab = load_content_vocab()
    pommel = {
        "type": "take_reward_card",
        "reward_choice": {"kind": "card", "content_id": "pommel_strike"},
        "card": {"card_id": "pommel_strike", "zone": "reward"},
    }
    shrug = {
        "type": "take_reward_card",
        "reward_choice": {"kind": "card", "content_id": "shrug_it_off"},
        "card": {"card_id": "shrug_it_off", "zone": "reward"},
    }

    pommel_ids = descriptor_identity_ids(pommel, vocab=vocab)
    shrug_ids = descriptor_identity_ids(shrug, vocab=vocab)

    assert descriptor_identity_tokens(pommel)[0] == "card:pommel_strike"
    assert descriptor_identity_tokens(shrug)[0] == "card:shrug_it_off"
    assert pommel_ids != shrug_ids
    assert pommel_ids[0] == vocab.id_for_token("card:pommel_strike")
    assert shrug_ids[0] == vocab.id_for_token("card:shrug_it_off")
    assert len(pommel_ids) == CONTENT_IDENTITY_SLOTS


def test_descriptor_identity_ids_include_event_option_identity() -> None:
    vocab = load_content_vocab()
    descriptor = {
        "type": "choose_event",
        "event_option": {
            "event_id": "ABYSSAL_BATHS",
            "page_id": "INITIAL",
            "option_id": "IMMERSE",
        },
    }

    tokens = descriptor_identity_tokens(descriptor)
    ids = descriptor_identity_ids(descriptor, vocab=vocab)

    assert "event_option:abyssal_baths:initial:immerse" in tokens
    assert "event:abyssal_baths" in tokens
    assert ids[0] == vocab.id_for_token("event_option:abyssal_baths:initial:immerse")

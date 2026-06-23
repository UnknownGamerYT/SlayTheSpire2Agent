from __future__ import annotations

import random

import pytest

from sts2sim.live_capture import LiveApiError
from sts2sim.live_start import _choose_ascension, _choose_character, _options


def test_choose_character_uses_only_unlocked_enabled_characters() -> None:
    state = {
        "characters": [
            {"id": "IRONCLAD", "name": "The Ironclad", "locked": False},
            {"id": "SILENT", "name": "The Silent", "locked": True},
            {"id": "DEFECT", "name": "The Defect", "locked": False},
        ],
        "options": [
            {"name": "IRONCLAD", "enabled": True},
            {"name": "SILENT", "enabled": False},
            {"name": "DEFECT", "enabled": False},
        ],
    }

    chosen = _choose_character(state, "random", random.Random("seed"))

    assert chosen["id"] == "IRONCLAD"


def test_choose_character_rejects_locked_requested_character() -> None:
    state = {
        "characters": [
            {"id": "IRONCLAD", "name": "The Ironclad", "locked": False},
            {"id": "SILENT", "name": "The Silent", "locked": True},
        ],
        "options": [
            {"name": "IRONCLAD", "enabled": True},
            {"name": "SILENT", "enabled": False},
        ],
    }

    with pytest.raises(LiveApiError):
        _choose_character(state, "SILENT", random.Random("seed"))


def test_choose_ascension_respects_reported_unlock_cap() -> None:
    state = {"ascension": 0, "max_ascension": 3}

    assert _choose_ascension(state, "max", random.Random("seed")) == 3
    assert _choose_ascension(state, "2", random.Random("seed")) == 2

    with pytest.raises(LiveApiError):
        _choose_ascension(state, "4", random.Random("seed"))


def test_options_accept_string_and_mapping_shapes() -> None:
    state = {
        "options": [
            "singleplayer",
            {"name": "standard", "enabled": True},
            {"name": "daily", "enabled": False},
        ]
    }

    assert _options(state) == [
        {"name": "singleplayer", "enabled": True},
        {"name": "standard", "enabled": True},
        {"name": "daily", "enabled": False},
    ]

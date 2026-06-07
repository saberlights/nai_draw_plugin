from __future__ import annotations

from core.reply_command_text import normalize_reply_command_text


def test_quoted_nai_command_is_not_current_command_text() -> None:
    message = {
        "raw_message": [
            {
                "type": "reply",
                "data": {
                    "target_message_id": "1816145810",
                    "target_message_content": "/nai selfie",
                },
            },
            {"type": "text", "data": "111"},
        ],
        "processed_plain_text": "/nai selfie 111",
    }

    assert normalize_reply_command_text(message) == "[reply]111"


def test_current_nai_command_after_reply_keeps_current_command() -> None:
    message = {
        "raw_message": [
            {
                "type": "reply",
                "data": {
                    "target_message_id": "1816145810",
                    "target_message_content": "/nai selfie",
                },
            },
            {"type": "text", "data": "/nai i2i forest background"},
        ],
        "processed_plain_text": "/nai selfie /nai i2i forest background",
    }

    assert normalize_reply_command_text(message) == "/nai i2i forest background"


def test_current_nai_command_after_non_command_reply_keeps_current_command() -> None:
    message = {
        "raw_message": [
            {
                "type": "reply",
                "data": {
                    "target_message_id": "1816145810",
                    "target_message_content": "previous normal message",
                },
            },
            {"type": "text", "data": "/nai selfie"},
        ],
        "processed_plain_text": "previous normal message /nai selfie",
    }

    assert normalize_reply_command_text(message) == "/nai selfie"


def test_non_nai_reply_is_not_rewritten() -> None:
    message = {
        "raw_message": [
            {
                "type": "reply",
                "data": {
                    "target_message_id": "1816145810",
                    "target_message_content": "previous normal message",
                },
            },
            {"type": "text", "data": "111"},
        ],
        "processed_plain_text": "previous normal message 111",
    }

    assert normalize_reply_command_text(message) is None

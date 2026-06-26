import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.services.session_state import session_state


def test_pending_image_generation_lifecycle() -> None:
    stream_id = "test_stream_pending_generation"

    session_state.clear_pending_image_generation(stream_id)
    assert session_state.get_pending_image_generation_started_at(stream_id) is None

    session_state.set_pending_image_generation(stream_id, started_at=123.0)
    assert session_state.get_pending_image_generation_started_at(stream_id) == 123.0

    session_state.clear_pending_image_generation(stream_id)
    assert session_state.get_pending_image_generation_started_at(stream_id) is None


def test_tag_retriever_show_toggle_uses_runtime_override() -> None:
    platform = "test-platform"
    chat_id = "test-chat-tag-show"

    session_state.clear_session_state(platform, chat_id)

    def _get_config(key: str, default=None):
        if key == "tag_retriever.show_result":
            return True
        return default

    assert session_state.is_tag_retriever_show_enabled(platform, chat_id, _get_config) is True

    session_state.set_tag_retriever_show_enabled(platform, chat_id, False)
    assert session_state.is_tag_retriever_show_enabled(platform, chat_id, _get_config) is False

    summary = session_state.get_session_state_summary(platform, chat_id)
    assert summary["tag_retriever_show"] is False

    session_state.clear_session_state(platform, chat_id)

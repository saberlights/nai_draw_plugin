import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.nai_draw_plugin.core.services.session_state import session_state


def test_pending_image_generation_lifecycle() -> None:
    stream_id = "test_stream_pending_generation"

    session_state.clear_pending_image_generation(stream_id)
    assert session_state.get_pending_image_generation_started_at(stream_id) is None

    started_at = time.time()
    session_state.set_pending_image_generation(stream_id, started_at=started_at)
    assert session_state.get_pending_image_generation_started_at(stream_id) == started_at

    session_state.clear_pending_image_generation(stream_id)
    assert session_state.get_pending_image_generation_started_at(stream_id) is None


def test_backend_selection_is_session_scoped() -> None:
    platform = "test-platform"
    chat_id = "test-chat-backend"
    session_state.clear_session_state(platform, chat_id)

    assert session_state.get_selected_backend(platform, chat_id) is None
    session_state.set_selected_backend(platform, chat_id, "chami")
    assert session_state.get_selected_backend(platform, chat_id) == "chami"

    session_state.clear_session_state(platform, chat_id)
    assert session_state.get_selected_backend(platform, chat_id) is None


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


def _retag_config_getter(default_path: str):
    def _get_config(key: str, default=None):
        if key == "retag.default_path":
            return default_path
        return default

    return _get_config


def test_retag_path_defaults_to_config_until_session_picks_one() -> None:
    platform = "test-platform"
    chat_id = "test-chat-retag-path"
    session_state.clear_session_state(platform, chat_id)

    get_config = _retag_config_getter("wd14")
    assert session_state.get_retag_path(platform, chat_id, get_config) == "wd14"

    # 粘性选定优先于配置默认，且切换后立刻生效
    assert session_state.set_retag_path(platform, chat_id, "llm") == "llm"
    assert session_state.get_retag_path(platform, chat_id, get_config) == "llm"

    session_state.clear_session_state(platform, chat_id)
    assert session_state.get_retag_path(platform, chat_id, get_config) == "wd14"


def test_retag_path_falls_back_to_auto_when_config_blank() -> None:
    platform = "test-platform"
    chat_id = "test-chat-retag-path-blank"
    session_state.clear_session_state(platform, chat_id)

    assert session_state.get_retag_path(platform, chat_id, _retag_config_getter("")) == "auto"


def test_retag_path_normalizes_case_and_whitespace() -> None:
    platform = "test-platform"
    chat_id = "test-chat-retag-path-normalize"
    session_state.clear_session_state(platform, chat_id)

    assert session_state.set_retag_path(platform, chat_id, "  LLM ") == "llm"
    assert session_state.get_retag_path(platform, chat_id, _retag_config_getter("auto")) == "llm"


def test_retag_path_rejects_unknown_value_without_mutating_state() -> None:
    """非法值必须抛出，且不能把会话原有选定改坏。"""
    platform = "test-platform"
    chat_id = "test-chat-retag-path-invalid"
    session_state.clear_session_state(platform, chat_id)
    session_state.set_retag_path(platform, chat_id, "meta")

    try:
        session_state.set_retag_path(platform, chat_id, "vlm")
    except ValueError as exc:
        assert "vlm" in str(exc)
    else:
        raise AssertionError("非法反推路径应抛 ValueError")

    assert session_state.get_retag_path(platform, chat_id, _retag_config_getter("auto")) == "meta"


def test_retag_path_config_typo_surfaces_instead_of_silent_auto() -> None:
    """配置里的 default_path 打错时要炸出来，不能悄悄按 auto 跑。"""
    platform = "test-platform"
    chat_id = "test-chat-retag-path-config-typo"
    session_state.clear_session_state(platform, chat_id)

    try:
        session_state.get_retag_path(platform, chat_id, _retag_config_getter("wd_14"))
    except ValueError as exc:
        assert "wd_14" in str(exc)
    else:
        raise AssertionError("非法 default_path 应抛 ValueError")


def test_retag_path_appears_in_session_summary() -> None:
    platform = "test-platform"
    chat_id = "test-chat-retag-path-summary"
    session_state.clear_session_state(platform, chat_id)

    assert session_state.get_session_state_summary(platform, chat_id)["retag_path"] is None

    session_state.set_retag_path(platform, chat_id, "llm")
    assert session_state.get_session_state_summary(platform, chat_id)["retag_path"] == "llm"

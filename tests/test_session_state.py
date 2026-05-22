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

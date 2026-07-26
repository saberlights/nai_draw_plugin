from pathlib import Path
import sys


MAIBOT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIBOT_ROOT))

from plugins.nai_draw_plugin.core.services.session_preferences import SessionPreferences


def test_preference_keys_use_the_same_normalization_for_write_read_and_clear() -> None:
    preferences = SessionPreferences()

    preferences.update(" stream:chat ", selected_model="nai-diffusion-4-5-full")

    assert preferences.get("stream:chat").selected_model == "nai-diffusion-4-5-full"
    preferences.clear(" stream:chat ")
    assert preferences.get("stream:chat") is None

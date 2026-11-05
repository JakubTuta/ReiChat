from __future__ import annotations

from app.chat import context_window as cw


def _msg(role: str, words: int, image: str = "") -> dict:
    return {"role": role, "content": "word " * words, "image": image}


def test_resolve_num_ctx_defaults_to_standard_and_caps_at_model_max():
    assert cw.resolve_num_ctx(None, None) == 8192
    assert cw.resolve_num_ctx(None, 4096) == 4096
    assert cw.resolve_num_ctx(32768, 131072) == 32768
    assert cw.resolve_num_ctx(32768, 16384) == 16384


def test_preset_for_maps_known_sizes_only():
    assert cw.preset_for(4096) == "short"
    assert cw.preset_for(16384) == "long"
    assert cw.preset_for(5000) is None


def test_short_history_fits_untouched():
    history = [_msg("user", 10), _msg("assistant", 10)]

    fit = cw.fit_history("Be nice.", history, _msg("user", 5), 4096)

    assert fit.dropped == 0
    assert fit.history == history
    assert fit.overflow is False


def test_oldest_messages_go_first_and_history_restarts_on_a_user_turn():
    history = [_msg("user" if i % 2 == 0 else "assistant", 500) for i in range(12)]

    fit = cw.fit_history(None, history, _msg("user", 5), 4096)

    assert fit.dropped > 0
    assert fit.history == history[fit.dropped :]
    assert fit.history[0]["role"] == "user"
    assert fit.estimated_prompt_tokens <= 4096 - cw.response_reserve(4096)


def test_old_images_are_dropped_before_any_message():
    history = [_msg("user", 10, image="data:image/png;base64,AAAA") for _ in range(4)]

    fit = cw.fit_history(None, history, _msg("user", 5), 4096)

    assert fit.dropped == 0
    assert fit.images_dropped >= 1
    assert fit.history[0]["image"] == ""
    # The caller's list is untouched — only the copy sent to the model changes.
    assert history[0]["image"]


def test_a_new_message_too_big_for_the_window_is_kept_and_flagged():
    fit = cw.fit_history(None, [_msg("user", 10)], _msg("user", 5000), 4096)

    assert fit.history == []
    assert fit.overflow is True


def test_summary_prompt_keeps_the_newest_part_of_an_oversized_transcript():
    messages = [{"role": "user", "content": f"msg{i} " + "x" * 1900} for i in range(40)]

    prompt = cw.build_summary_prompt("", messages, 4096)

    assert "msg39" in prompt
    assert "msg0 " not in prompt
    assert len(prompt) <= (4096 - cw.response_reserve(4096)) * cw.CHARS_PER_TOKEN

from __future__ import annotations

import pytest

from app.db import models
from app.openai_compat.parsing import OpenAICompatError, parse_chat_completion_request

pytestmark = pytest.mark.usefixtures("db")


@pytest.fixture
async def ai_model(db):
    model = models.AIModel(name="llama3.1", model="llama3.1", description="", index=1)
    db.add(model)
    await db.commit()
    return model


@pytest.fixture
def parse(db):
    async def run(body):
        return await parse_chat_completion_request(body, db)

    return run


async def test_parses_minimal_request(ai_model, parse):
    parsed = await parse({"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hi"}]})

    assert parsed["ai_model"] == ai_model
    assert parsed["parameters"] == "8b"
    assert parsed["message"] == "Hi"
    assert parsed["image"] == ""
    assert parsed["history"] == []
    assert parsed["system_prompt"] is None
    assert parsed["stream"] is False
    assert parsed["generation_params"] == {}


async def test_extracts_system_message_and_history(ai_model, parse):
    parsed = await parse(
        {
            "model": "llama3.1:8b",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ],
        }
    )

    assert parsed["system_prompt"] == "Be terse."
    assert parsed["message"] == "How are you?"
    assert parsed["history"] == [
        {"role": "user", "content": "Hi", "image": ""},
        {"role": "assistant", "content": "Hello", "image": ""},
    ]


async def test_extracts_text_and_image_from_content_parts(ai_model, parse):
    parsed = await parse(
        {
            "model": "llama3.1:8b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }
            ],
        }
    )

    assert parsed["message"] == "What's this?"
    assert parsed["image"] == "data:image/png;base64,abc"


async def test_parses_generation_params(ai_model, parse):
    parsed = await parse(
        {
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "seed": 42,
        }
    )

    assert parsed["generation_params"] == {"temperature": 0.5, "top_p": 0.9, "seed": 42}


async def test_unknown_model_raises_404_with_model_not_found_code(ai_model, parse):
    with pytest.raises(OpenAICompatError) as exc_info:
        await parse({"model": "does-not-exist:8b", "messages": [{"role": "user", "content": "Hi"}]})

    assert exc_info.value.status == 404
    assert exc_info.value.code == "model_not_found"


async def test_last_message_not_user_raises(ai_model, parse):
    with pytest.raises(OpenAICompatError) as exc_info:
        await parse(
            {
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            }
        )

    assert exc_info.value.status == 400


async def test_invalid_role_raises(ai_model, parse):
    with pytest.raises(OpenAICompatError) as exc_info:
        await parse({"model": "llama3.1:8b", "messages": [{"role": "tool", "content": "42"}]})

    assert exc_info.value.status == 400
    assert exc_info.value.param == "messages"


async def test_tools_param_is_rejected(ai_model, parse):
    with pytest.raises(OpenAICompatError) as exc_info:
        await parse(
            {
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{"type": "function", "function": {"name": "foo"}}],
            }
        )

    assert exc_info.value.status == 400
    assert exc_info.value.param == "tools"

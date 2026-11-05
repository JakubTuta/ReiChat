"""What the chat sockets accept from the browser, checked before any model or
database work starts. Every message here is shown to the user as is."""

import base64
import binascii
import typing

MAX_MESSAGE_LENGTH = 10_000
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # measured on the decoded binary
MAX_STRUCTURED_FIELD_DESCRIPTION_LENGTH = 500

INVALID_TARGET_MESSAGE = "Invalid model or chat history."


class InvalidMessage(ValueError):
    """The client sent something the socket can't act on; the text says what to fix."""


def require_keys(data: dict[str, typing.Any], *keys: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise InvalidMessage(f"The message is missing required fields: {', '.join(missing)}.")


def check_text(value: object, what: str) -> str:
    if not value or not isinstance(value, str):
        raise InvalidMessage(f"{what} must be a non-empty string.")
    if len(value) > MAX_MESSAGE_LENGTH:
        raise InvalidMessage(f"{what} too long. Maximum length is {MAX_MESSAGE_LENGTH} characters.")
    return value


def check_model_name(value: object) -> str:
    if not isinstance(value, str):
        raise InvalidMessage(INVALID_TARGET_MESSAGE)
    return value


def check_parameters(value: object) -> str:
    if not isinstance(value, str):
        raise InvalidMessage("ai_model_parameters must be a string.")
    return value


def check_index(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidMessage("index must be an integer.")
    return value


def check_image(image: object) -> str:
    if not image:
        return ""

    if not isinstance(image, str) or not image.startswith("data:image/"):
        raise InvalidMessage("Invalid image format. Expected a base64 data URI.")

    image_parts = image.split(",", 1)
    if len(image_parts) < 2 or not image_parts[1]:
        raise InvalidMessage("Invalid image format. Expected a base64 data URI.")

    try:
        decoded_size = len(base64.b64decode(image_parts[1], validate=True))
    except (binascii.Error, ValueError) as e:
        raise InvalidMessage("Invalid image data — could not decode base64.") from e

    if decoded_size > MAX_IMAGE_SIZE_BYTES:
        raise InvalidMessage(
            f"Image exceeds the maximum allowed size of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB."
        )
    return image


def check_structured_output(structured_output: object) -> list[dict[str, str | None]] | None:
    if structured_output is None:
        return None

    if not isinstance(structured_output, list):
        raise InvalidMessage("structured_output must be a list.")

    for i, spec in enumerate(structured_output):
        if not isinstance(spec, dict):
            raise InvalidMessage(f"structured_output[{i}] must be an object.")
        if not isinstance(spec.get("field"), str) or not spec["field"].strip():
            raise InvalidMessage(f"structured_output[{i}].field must be a non-empty string.")
        if not isinstance(spec.get("type"), str) or not spec["type"].strip():
            raise InvalidMessage(f"structured_output[{i}].type must be a non-empty string.")

        description = spec.get("description")
        if description is not None and (
            not isinstance(description, str) or len(description) > MAX_STRUCTURED_FIELD_DESCRIPTION_LENGTH
        ):
            raise InvalidMessage(
                f"structured_output[{i}].description must be a string of at most "
                f"{MAX_STRUCTURED_FIELD_DESCRIPTION_LENGTH} characters."
            )

        array_type = spec.get("arrayType")
        if array_type is not None and not isinstance(array_type, str):
            raise InvalidMessage(f"structured_output[{i}].arrayType must be a string.")

    return typing.cast("list[dict[str, str | None]]", structured_output)

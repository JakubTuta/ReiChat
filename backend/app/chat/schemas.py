import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from app.catalog.schemas import AIModelOut
from app.core.types import NonBlankStr, reject_explicit_nulls
from app.db.models import DEFAULT_CHAT_TITLE

_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1


class PersonaRef(BaseModel):
    id: int
    name: str


class CollectionRef(BaseModel):
    id: int
    name: str


class ChatRef(BaseModel):
    id: int
    title: str


class ChatListItem(BaseModel):
    """What the chat list needs about each chat — settings, not messages."""

    id: int
    title: str
    persona: PersonaRef | None
    temperature: float | None
    num_ctx: int | None
    top_p: float | None
    seed: int | None
    active_collections: list[CollectionRef]
    tools_enabled: bool
    memory_mode: str
    think_level: str
    continued_from: ChatRef | None


class ChatCreate(BaseModel):
    title: NonBlankStr = DEFAULT_CHAT_TITLE


class ChatCreatedOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ai_model: AIModelOut
    title: str
    last_update_time: datetime.datetime


class ChatRemove(BaseModel):
    chat_id: int


def _whole_number(value: Any) -> Any:
    # A number the UI sent as 4096.7 is cut to 4096, as it always was; anything
    # that isn't a number at all (text, true/false) is refused.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("must be a number or null")
    try:
        return int(value)
    except (ValueError, OverflowError):
        raise ValueError("must be a finite number") from None


WholeNumber = Annotated[int, BeforeValidator(_whole_number), Field(ge=_INT32_MIN, le=_INT32_MAX)]

# The sane UI bounds for a slider. Ollama doesn't itself enforce these ranges,
# but they catch obvious mistakes (e.g. temperature=200). Below ~1k tokens there
# is no room for even one exchange plus the reply.
Temperature = Annotated[StrictFloat, Field(ge=0.0, le=2.0)]
TopP = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
ContextSize = Annotated[WholeNumber, Field(ge=1024)]

# What a chat update may clear by sending null.
_CLEARABLE = frozenset({"persona_id", "temperature", "num_ctx", "top_p", "seed"})
UPDATABLE_CHAT_FIELDS = frozenset(
    {
        "title",
        "persona_id",
        "collection_ids",
        "tools_enabled",
        "memory_mode",
        "think_level",
        "move_to_model",
        "temperature",
        "num_ctx",
        "top_p",
        "seed",
    }
)


class ChatUpdate(BaseModel):
    """Only the fields that are sent change; a field left out is left alone."""

    id: int
    title: str | None = None
    persona_id: int | None = None
    collection_ids: list[StrictInt] | None = None
    tools_enabled: StrictBool | None = None
    # Continue this conversation with a different model: the chat moves to that
    # model's list; earlier replies keep their own model_label, so nothing about
    # who wrote what is lost.
    move_to_model: str | None = None
    # Free-form past "off"/"on" on purpose: level names come from the model
    # itself (/api/show), and resolve_reasoning ignores one the model doesn't list.
    think_level: Annotated[str, Field(max_length=32)] | None = None
    memory_mode: Literal["trim", "summarize"] | None = None
    temperature: Temperature | None = None
    num_ctx: ContextSize | None = None
    top_p: TopP | None = None
    seed: WholeNumber | None = None

    @model_validator(mode="after")
    def _only_sent_fields_may_change(self) -> "ChatUpdate":
        return reject_explicit_nulls(self, nullable=_CLEARABLE)


class BranchSwitch(BaseModel):
    # Positions in the active path, as the UI counts them: text or true/false would be a bug there.
    index: StrictInt
    sibling_index: StrictInt


class ChatContinue(BaseModel):
    chat_id: int


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("can't be blank")
    return value


# Checked for being non-blank but stored as written.
NotBlank = Annotated[str, AfterValidator(_not_blank)]


class ChatFromCompare(BaseModel):
    parameters: NotBlank
    prompt: NotBlank
    response: NotBlank

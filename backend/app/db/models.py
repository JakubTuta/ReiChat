import datetime
from typing import Annotated, Any, ClassVar

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Table,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

DEFAULT_CHAT_TITLE = "New chat"

IntPK = Annotated[int, mapped_column(BigInteger, Identity(), primary_key=True)]


def utcnow() -> datetime.datetime:
    # Python-side rather than the database's now(): that is fixed per
    # transaction, which would give a user message and its reply one timestamp
    # and make their order ambiguous.
    return datetime.datetime.now(datetime.UTC)


class AIModel(Base):
    __tablename__ = "ai_model"

    id: Mapped[IntPK]
    name: Mapped[str]
    model: Mapped[str] = mapped_column(unique=True)
    description: Mapped[str]
    popularity: Mapped[int] = mapped_column(Integer, default=0)
    can_process_image: Mapped[bool] = mapped_column(Boolean, default=False)
    # True for embedding-only models (e.g. nomic-embed-text). Excluded from the
    # chat model picker; used for RAG.
    is_embedding: Mapped[bool] = mapped_column(Boolean, default=False)
    # True when ollama.com lists the 'tools' capability chip for this model —
    # gates whether the tool-calling toggle is offered for it.
    can_use_tools: Mapped[bool] = mapped_column(Boolean, default=False)
    index: Mapped[int] = mapped_column(Integer, default=0)
    # Added by the user by name (any Ollama tag, e.g. a Hugging Face GGUF),
    # not scraped from ollama.com — so catalog refreshes leave it alone.
    is_custom: Mapped[bool] = mapped_column(Boolean, default=False)

    versions: Mapped[list["AIModelVersion"]] = relationship(
        back_populates="ai_model", lazy="raise", order_by="AIModelVersion.id", passive_deletes=True
    )


class AIModelVersion(Base):
    __tablename__ = "ai_model_version"
    __table_args__ = (UniqueConstraint("ai_model_id", "parameters", name="unique_ai_model_version"),)

    id: Mapped[IntPK]
    ai_model_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("ai_model.id", ondelete="CASCADE"))
    parameters: Mapped[str]
    size: Mapped[str]
    # Exact pull size from the registry manifest.
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    # Last time this tag was confirmed pullable against registry.ollama.ai.
    verified_at: Mapped[datetime.datetime | None]

    ai_model: Mapped[AIModel] = relationship(back_populates="versions", lazy="raise")


class Persona(Base):
    __tablename__ = "persona"

    id: Mapped[IntPK]
    name: Mapped[str]
    system_prompt: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)


class PromptTemplate(Base):
    """A reusable starter message, not a system prompt (see Persona for that).
    `content` may contain `{{variable}}` placeholders — the frontend extracts
    them, prompts the user to fill each one in, and substitutes before dropping
    the result into the composer. Substitution is a pure frontend concern (see
    frontend/utils/promptTemplate.ts); this table just stores the template text.
    """

    __tablename__ = "prompt_template"

    id: Mapped[IntPK]
    name: Mapped[str]
    description: Mapped[str] = mapped_column(default="")
    content: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)


chat_history_active_collection = Table(
    "chat_history_active_collection",
    Base.metadata,
    Column(
        "chat_history_id", BigInteger, ForeignKey("chat_history.id", ondelete="CASCADE"), primary_key=True
    ),
    Column(
        "document_collection_id",
        BigInteger,
        ForeignKey("document_collection.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    # Deleting a collection looks its rows up by this side.
    Index("ix_chat_history_active_collection_document_collection_id", "document_collection_id"),
)


class ChatHistory(Base):
    __tablename__ = "chat_history"

    MEMORY_MODE_TRIM: ClassVar[str] = "trim"
    MEMORY_MODE_SUMMARIZE: ClassVar[str] = "summarize"

    id: Mapped[IntPK]
    ai_model_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ai_model.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(default=DEFAULT_CHAT_TITLE)
    # Refreshed by hand in exactly the writes that add or replace messages: an
    # onupdate hook would also fire on a rename or a branch switch and make the
    # chat jump to the top of the list.
    last_update_time: Mapped[datetime.datetime] = mapped_column(default=utcnow, index=True)
    persona_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("persona.id", ondelete="SET NULL"), index=True
    )
    # Per-chat ChatOllama generation parameters. Null means "use Ollama's own
    # default" — these are genuinely optional overrides, not settings with a
    # meaningful zero value.
    temperature: Mapped[float | None] = mapped_column(Float)
    num_ctx: Mapped[int | None] = mapped_column(Integer)
    top_p: Mapped[float | None] = mapped_column(Float)
    seed: Mapped[int | None] = mapped_column(Integer)
    # The tip of whichever branch is currently shown/continued. Walking up
    # from here via ChatMessage.parent_id reconstructs the active conversation —
    # regenerate/edit-resend just move this pointer to a new leaf instead of
    # deleting anything, so old branches always stay in the database.
    active_leaf_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("chat_message.id", ondelete="SET NULL", use_alter=True),
        index=True,
    )
    # Only takes effect when ai_model.can_use_tools is also true — this flag
    # alone doesn't mean tool calls will actually happen, just that they're
    # allowed for this chat once the model supports them.
    tools_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # What happens once the conversation outgrows the context window — see
    # chat/context_window.py.
    memory_mode: Mapped[str] = mapped_column(default=MEMORY_MODE_TRIM)
    # Running summary of the oldest messages, in summarize mode. Valid only
    # while memory_summary_until_id (the last message it covers) is still on the
    # active path — switching to a branch that diverged earlier discards it.
    memory_summary: Mapped[str] = mapped_column(default="")
    memory_summary_until_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("chat_message.id", ondelete="SET NULL", use_alter=True),
        index=True,
    )
    # "Start fresh with a summary": the chat this one continues. Its summary
    # is generated on this chat's first message (it can take a while on CPU,
    # longer than an HTTP request should) and then kept in seed_summary.
    seed_source_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("chat_history.id", ondelete="SET NULL"), index=True
    )
    seed_summary: Mapped[str] = mapped_column(default="")
    # "" = the model's own default; "off"/"on"; or a level name the model
    # lists (e.g. "low"). See llm/runtime.resolve_reasoning.
    think_level: Mapped[str] = mapped_column(default="")

    ai_model: Mapped[AIModel] = relationship(lazy="raise")
    persona: Mapped[Persona | None] = relationship(lazy="raise")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="chat",
        foreign_keys="ChatMessage.chat_id",
        lazy="raise",
        order_by="ChatMessage.created_at, ChatMessage.id",
        passive_deletes=True,
    )
    # Which document collections this chat retrieves from. A collection
    # created for a specific chat (DocumentCollection.chat_id) still has to be
    # added here explicitly — the FK is just "where it came from", this is
    # "what's actually turned on".
    active_collections: Mapped[list["DocumentCollection"]] = relationship(
        secondary=chat_history_active_collection, lazy="raise", order_by="DocumentCollection.id"
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"
    __table_args__ = (
        Index("chatmessage_chat_created_idx", "chat_id", "created_at"),
        Index("chatmessage_parent_idx", "parent_id"),
    )

    id: Mapped[IntPK]
    chat_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chat_history.id", ondelete="CASCADE"))
    parent_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("chat_message.id", ondelete="CASCADE")
    )
    role: Mapped[str]
    content: Mapped[str]
    image: Mapped[str] = mapped_column(default="")
    # A thinking model's reasoning for this reply, kept apart from the answer.
    thinking: Mapped[str] = mapped_column(default="")
    # "name:tag" of the model that wrote an assistant reply — a chat can be
    # continued with a different model, so the chat's own model isn't
    # enough to tell who said what.
    model_label: Mapped[str] = mapped_column(default="")
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)

    chat: Mapped[ChatHistory] = relationship(back_populates="messages", foreign_keys=[chat_id], lazy="raise")


class AppSetting(Base):
    """A setting the user changes from the UI — see runtime_settings.py."""

    __tablename__ = "app_setting"

    id: Mapped[IntPK]
    key: Mapped[str] = mapped_column(unique=True)
    value: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=True)


class CompareVote(Base):
    """One blind comparison the user judged: which answer they liked best
    among the models that answered the same prompt. Feeds the personal
    scoreboard on the Compare page."""

    __tablename__ = "compare_vote"

    id: Mapped[IntPK]
    prompt: Mapped[str]
    winner: Mapped[str]
    # "name:tag" of every model that answered, winner included.
    contenders: Mapped[list[str]] = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)


class MCPServer(Base):
    """An external Model Context Protocol server the user has configured —
    its tools are merged with the built-in ones whenever tool calling is on
    for a chat. No per-chat scoping: unlike documents, an MCP server is a
    system-level integration, not per-conversation data.
    """

    __tablename__ = "mcp_server"

    TRANSPORT_STDIO: ClassVar[str] = "stdio"
    TRANSPORT_HTTP: ClassVar[str] = "http"

    id: Mapped[IntPK]
    name: Mapped[str] = mapped_column(unique=True)
    transport: Mapped[str]
    # stdio: a shell-style command line (parsed with shlex, not a real
    # shell — no pipes/redirection). http: a URL. Exactly one is used,
    # depending on `transport`; the other stays blank.
    command: Mapped[str] = mapped_column(default="")
    url: Mapped[str] = mapped_column(default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Off by default: an MCP tool can do anything its server can (send mail,
    # delete files), and the model can be steered into calling one by text
    # it reads — a web page, an attached document. So each call is shown to
    # the user to allow or deny, unless they've said this server is trusted.
    auto_approve: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)


class DocumentCollection(Base):
    __tablename__ = "document_collection"

    id: Mapped[IntPK]
    name: Mapped[str]
    # Null => a global collection, attachable to any chat. Set => created
    # from a specific chat's document panel, but still just a regular row
    # any chat can attach via ChatHistory.active_collections — this FK is
    # provenance, not an access restriction.
    chat_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("chat_history.id", ondelete="CASCADE"), index=True
    )
    # Every document in this collection is chunked and embedded with this
    # exact model+parameters, so all of its chunk vectors share one
    # dimensionality and are safe to compare with cosine distance. Querying
    # this collection later re-embeds the search query with the same model.
    # RESTRICT: removing the model would orphan every vector in the collection.
    embedding_model_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ai_model.id", ondelete="RESTRICT"), index=True
    )
    embedding_parameters: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)

    embedding_model: Mapped[AIModel] = relationship(lazy="raise")
    documents: Mapped[list["Document"]] = relationship(
        back_populates="collection", lazy="raise", order_by="Document.id", passive_deletes=True
    )


class Document(Base):
    __tablename__ = "document"

    STATUS_PENDING: ClassVar[str] = "pending"
    STATUS_PROCESSING: ClassVar[str] = "processing"
    STATUS_READY: ClassVar[str] = "ready"
    STATUS_FAILED: ClassVar[str] = "failed"

    id: Mapped[IntPK]
    collection_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("document_collection.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str]
    status: Mapped[str] = mapped_column(default=STATUS_PENDING)
    error_message: Mapped[str] = mapped_column(default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    # Percent of chunks embedded so far, while processing — a big PDF on a
    # CPU-only machine can take minutes.
    progress: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)

    collection: Mapped[DocumentCollection] = relationship(back_populates="documents", lazy="raise")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", lazy="raise", order_by="DocumentChunk.chunk_index", passive_deletes=True
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunk"

    id: Mapped[IntPK]
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str]
    char_start: Mapped[int] = mapped_column(Integer)
    # 1-based page the chunk starts on, for PDFs; null for formats without pages.
    page: Mapped[int | None] = mapped_column(Integer)
    # No fixed dimension — different collections may use different embedding
    # models with different vector sizes. Safe because every cosine-distance
    # query is scoped to chunks from one collection (see rag/retrieval.py), so
    # it only ever compares same-dimension vectors.
    embedding: Mapped[list[float]] = mapped_column(VECTOR())

    document: Mapped[Document] = relationship(back_populates="chunks", lazy="raise")


class ContainerPort(Base):
    """Stable host-port allocation for an Ollama container.

    Replaces `11434 + ai_model.index`: `index` is renumbered on every catalog
    refresh and has no uniqueness constraint, so two different models could
    compute the same host port. When that happened, closing whatever listened
    on the port would stop an already-running container — silently killing an
    unrelated live chat. A port allocated here is never reused for a different
    container.
    """

    __tablename__ = "container_port"

    id: Mapped[IntPK]
    container_name: Mapped[str] = mapped_column(unique=True)
    port: Mapped[int] = mapped_column(Integer, unique=True)
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)

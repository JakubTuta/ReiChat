from django.db import models
from pgvector.django import VectorField


class AIModel(models.Model):
    name = models.TextField()
    model = models.TextField(unique=True)
    description = models.TextField()
    popularity = models.IntegerField(default=0)
    can_process_image = models.BooleanField(default=False)
    is_embedding = models.BooleanField(
        default=False,
        help_text="True for embedding-only models (e.g. nomic-embed-text). "
        "Excluded from the chat model picker; used for RAG.",
    )
    can_use_tools = models.BooleanField(
        default=False,
        help_text="True when ollama.com lists the 'tools' capability chip for this model — "
        "gates whether the tool-calling toggle is offered for it.",
    )
    index = models.IntegerField(default=0)

    def __str__(self) -> str:
        return self.model


class AIModelVersion(models.Model):
    ai_model = models.ForeignKey(AIModel, on_delete=models.CASCADE, related_name="versions")
    parameters = models.TextField()
    size = models.TextField()
    size_bytes = models.BigIntegerField(
        null=True, blank=True, help_text="Exact pull size from the registry manifest."
    )
    verified_at = models.DateTimeField(
        null=True, blank=True, help_text="Last time this tag was confirmed pullable against registry.ollama.ai."
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["ai_model", "parameters"], name="unique_ai_model_version"),
        ]

    def __str__(self) -> str:
        return f"{self.ai_model.model}:{self.parameters}"


class Persona(models.Model):
    name = models.TextField()
    system_prompt = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name


class PromptTemplate(models.Model):
    """A reusable starter message, not a system prompt (see Persona for
    that). `content` may contain `{{variable}}` placeholders — the frontend
    extracts them, prompts the user to fill each one in, and substitutes
    before dropping the result into the composer. Substitution is a pure
    frontend concern (see frontend/utils/promptTemplate.ts); this model just
    stores the template text.
    """

    name = models.TextField()
    description = models.TextField(default="", blank=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name


DEFAULT_CHAT_TITLE = "New chat"


class ChatHistory(models.Model):
    ai_model = models.ForeignKey(AIModel, on_delete=models.CASCADE)
    title = models.TextField(default=DEFAULT_CHAT_TITLE)
    last_update_time = models.DateTimeField(auto_now=True, db_index=True)
    persona = models.ForeignKey(
        Persona, null=True, blank=True, on_delete=models.SET_NULL, related_name="chats"
    )
    # Per-chat ChatOllama generation parameters. Null means "use Ollama's own
    # default" — these are genuinely optional overrides, not settings with a
    # meaningful zero value.
    temperature = models.FloatField(null=True, blank=True)
    num_ctx = models.IntegerField(null=True, blank=True)
    top_p = models.FloatField(null=True, blank=True)
    seed = models.IntegerField(null=True, blank=True)
    # The tip of whichever branch is currently shown/continued. Walking up
    # from here via ChatMessage.parent reconstructs the active conversation —
    # regenerate/edit-resend just move this pointer to a new leaf instead of
    # deleting anything, so old branches always stay in the database.
    active_leaf = models.ForeignKey(
        "ChatMessage", null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    # Which document collections this chat retrieves from. A collection
    # created for a specific chat (DocumentCollection.chat) still has to be
    # added here explicitly — the FK is just "where it came from", this is
    # "what's actually turned on".
    active_collections = models.ManyToManyField(
        "DocumentCollection", blank=True, related_name="active_in_chats"
    )
    # Only takes effect when ai_model.can_use_tools is also true — this flag
    # alone doesn't mean tool calls will actually happen, just that they're
    # allowed for this chat once the model supports them.
    tools_enabled = models.BooleanField(default=False)

    MEMORY_MODE_TRIM = "trim"
    MEMORY_MODE_SUMMARIZE = "summarize"
    MEMORY_MODE_CHOICES = [
        (MEMORY_MODE_TRIM, "Forget the oldest messages"),
        (MEMORY_MODE_SUMMARIZE, "Summarize older messages"),
    ]
    # What happens once the conversation outgrows the context window — see
    # context_window.py.
    memory_mode = models.TextField(choices=MEMORY_MODE_CHOICES, default=MEMORY_MODE_TRIM)
    # Running summary of the oldest messages, in summarize mode. Valid only
    # while memory_summary_until (the last message it covers) is still on the
    # active path — switching to a branch that diverged earlier discards it.
    memory_summary = models.TextField(default="", blank=True)
    memory_summary_until = models.ForeignKey(
        "ChatMessage", null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    # "Start fresh with a summary": the chat this one continues. Its summary
    # is generated on this chat's first message (it can take a while on CPU,
    # longer than an HTTP request should) and then kept in seed_summary.
    seed_source = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL, related_name="continuations"
    )
    seed_summary = models.TextField(default="", blank=True)
    # "" = the model's own default; "off"/"on"; or a level name the model
    # lists (e.g. "low"). See functions.resolve_reasoning.
    think_level = models.TextField(default="", blank=True)

    def __str__(self) -> str:
        return self.title


class ChatMessage(models.Model):
    chat = models.ForeignKey(ChatHistory, on_delete=models.CASCADE, related_name="messages")
    parent = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.CASCADE, related_name="children"
    )
    role = models.TextField()
    content = models.TextField()
    image = models.TextField(default="")
    # A thinking model's reasoning for this reply, kept apart from the answer.
    thinking = models.TextField(default="", blank=True)
    # "name:tag" of the model that wrote an assistant reply — a chat can be
    # continued with a different model, so the chat's own model isn't
    # enough to tell who said what.
    model_label = models.TextField(default="", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["chat", "created_at"], name="chatmessage_chat_created_idx"),
            models.Index(fields=["parent"], name="chatmessage_parent_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}"


class MCPServer(models.Model):
    """An external Model Context Protocol server the user has configured —
    its tools are merged with the built-in ones whenever tool calling is on
    for a chat (see rag/retrieval.py's collections for the same "global,
    opt-in via enabled" pattern). No per-chat scoping: unlike documents,
    an MCP server is a system-level integration, not per-conversation data.
    """

    TRANSPORT_STDIO = "stdio"
    TRANSPORT_HTTP = "http"
    TRANSPORT_CHOICES = [
        (TRANSPORT_STDIO, "stdio (local command)"),
        (TRANSPORT_HTTP, "HTTP (streamable)"),
    ]

    name = models.TextField(unique=True)
    transport = models.TextField(choices=TRANSPORT_CHOICES)
    # stdio: a shell-style command line (parsed with shlex, not a real
    # shell — no pipes/redirection). http: a URL. Exactly one is used,
    # depending on `transport`; the other stays blank.
    command = models.TextField(default="")
    url = models.TextField(default="")
    enabled = models.BooleanField(default=True)
    # Off by default: an MCP tool can do anything its server can (send mail,
    # delete files), and the model can be steered into calling one by text
    # it reads — a web page, an attached document. So each call is shown to
    # the user to allow or deny, unless they've said this server is trusted.
    auto_approve = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name


class DocumentCollection(models.Model):
    name = models.TextField()
    # Null => a global collection, attachable to any chat. Set => created
    # from a specific chat's document panel, but still just a regular row
    # any chat can attach via ChatHistory.active_collections — this FK is
    # provenance, not an access restriction.
    chat = models.ForeignKey(
        ChatHistory, null=True, blank=True, on_delete=models.CASCADE, related_name="owned_collections"
    )
    # Every document in this collection is chunked and embedded with this
    # exact model+parameters, so all of its chunk vectors share one
    # dimensionality and are safe to compare with CosineDistance. Querying
    # this collection later re-embeds the search query with the same model.
    embedding_model = models.ForeignKey(AIModel, on_delete=models.PROTECT, related_name="+")
    embedding_parameters = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name


class Document(models.Model):
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_READY = "ready"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_READY, "Ready"),
        (STATUS_FAILED, "Failed"),
    ]

    collection = models.ForeignKey(DocumentCollection, on_delete=models.CASCADE, related_name="documents")
    filename = models.TextField()
    status = models.TextField(choices=STATUS_CHOICES, default=STATUS_PENDING)
    error_message = models.TextField(default="")
    chunk_count = models.IntegerField(default=0)
    # Percent of chunks embedded so far, while processing — a big PDF on a
    # CPU-only machine can take minutes.
    progress = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.filename


class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.IntegerField()
    content = models.TextField()
    char_start = models.IntegerField()
    # 1-based page the chunk starts on, for PDFs; null for formats without pages.
    page = models.IntegerField(null=True, blank=True)
    # No fixed `dimensions=` — different collections may use different
    # embedding models with different vector sizes. Safe because every
    # CosineDistance query is scoped to chunks from one collection (see
    # rag/retrieval.py), so it only ever compares same-dimension vectors.
    embedding = VectorField()

    class Meta:
        ordering = ["chunk_index"]

    def __str__(self) -> str:
        return f"{self.document.filename}#{self.chunk_index}"

"""initial schema and the bundled model catalog

Revision ID: 0001
Revises:
Create Date: 2026-11-05 16:18:33.352834

"""

import json
import pathlib
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


SEED_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "model_catalog.json"


def seed_model_catalog() -> None:
    """Populate a fresh install with a working catalog and zero network calls,
    so the app is usable in the first five seconds instead of demanding a
    "Refresh model list" round-trip to ollama.com before a user can do anything.

    Self-contained on purpose: a migration must keep meaning what it meant when
    it was written, so it declares its own tables instead of importing the app's.
    """
    ai_model = sa.table(
        "ai_model",
        sa.column("id", sa.BigInteger),
        sa.column("name", sa.Text),
        sa.column("model", sa.Text),
        sa.column("description", sa.Text),
        sa.column("popularity", sa.Integer),
        sa.column("can_process_image", sa.Boolean),
        sa.column("is_embedding", sa.Boolean),
        sa.column("can_use_tools", sa.Boolean),
        sa.column("index", sa.Integer),
        sa.column("is_custom", sa.Boolean),
    )
    ai_model_version = sa.table(
        "ai_model_version",
        sa.column("ai_model_id", sa.BigInteger),
        sa.column("parameters", sa.Text),
        sa.column("size", sa.Text),
        sa.column("size_bytes", sa.BigInteger),
    )

    with open(SEED_PATH, encoding="utf-8") as f:
        seed_models = json.load(f)

    connection = op.get_bind()
    for position, entry in enumerate(seed_models, start=1):
        # No explicit id: the identity sequence has to stay in step with the rows.
        model_id = connection.execute(
            sa.insert(ai_model)
            .values(
                name=entry["name"],
                model=entry["model"],
                description=entry["description"],
                popularity=entry["popularity"],
                can_process_image=entry["can_process_image"],
                is_embedding=entry["is_embedding"],
                can_use_tools=False,
                index=position,
                is_custom=False,
            )
            .returning(ai_model.c.id)
        ).scalar_one()
        op.bulk_insert(
            ai_model_version,
            [
                {
                    "ai_model_id": model_id,
                    "parameters": version["parameters"],
                    "size": version["size"],
                    "size_bytes": version["size_bytes"],
                }
                for version in entry["versions"]
            ],
        )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "ai_model",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("popularity", sa.Integer(), nullable=False),
        sa.Column("can_process_image", sa.Boolean(), nullable=False),
        sa.Column("is_embedding", sa.Boolean(), nullable=False),
        sa.Column("can_use_tools", sa.Boolean(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("is_custom", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ai_model")),
        sa.UniqueConstraint("model", name=op.f("uq_ai_model_model")),
    )
    op.create_table(
        "app_setting",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_app_setting")),
        sa.UniqueConstraint("key", name=op.f("uq_app_setting_key")),
    )
    op.create_table(
        "compare_vote",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("winner", sa.Text(), nullable=False),
        sa.Column("contenders", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_compare_vote")),
    )
    op.create_table(
        "container_port",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("container_name", sa.Text(), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_container_port")),
        sa.UniqueConstraint("container_name", name=op.f("uq_container_port_container_name")),
        sa.UniqueConstraint("port", name=op.f("uq_container_port_port")),
    )
    op.create_table(
        "mcp_server",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("transport", sa.Text(), nullable=False),
        sa.Column("command", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("auto_approve", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mcp_server")),
        sa.UniqueConstraint("name", name=op.f("uq_mcp_server_name")),
    )
    op.create_table(
        "persona",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_persona")),
    )
    op.create_table(
        "prompt_template",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_prompt_template")),
    )
    op.create_table(
        "ai_model_version",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("ai_model_id", sa.BigInteger(), nullable=False),
        sa.Column("parameters", sa.Text(), nullable=False),
        sa.Column("size", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["ai_model_id"],
            ["ai_model.id"],
            name=op.f("fk_ai_model_version_ai_model_id_ai_model"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ai_model_version")),
        sa.UniqueConstraint("ai_model_id", "parameters", name="unique_ai_model_version"),
    )
    op.create_table(
        "chat_history",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("ai_model_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("last_update_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("persona_id", sa.BigInteger(), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("num_ctx", sa.Integer(), nullable=True),
        sa.Column("top_p", sa.Float(), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("active_leaf_id", sa.BigInteger(), nullable=True),
        sa.Column("tools_enabled", sa.Boolean(), nullable=False),
        sa.Column("memory_mode", sa.Text(), nullable=False),
        sa.Column("memory_summary", sa.Text(), nullable=False),
        sa.Column("memory_summary_until_id", sa.BigInteger(), nullable=True),
        sa.Column("seed_source_id", sa.BigInteger(), nullable=True),
        sa.Column("seed_summary", sa.Text(), nullable=False),
        sa.Column("think_level", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["ai_model_id"],
            ["ai_model.id"],
            name=op.f("fk_chat_history_ai_model_id_ai_model"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["persona_id"],
            ["persona.id"],
            name=op.f("fk_chat_history_persona_id_persona"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["seed_source_id"],
            ["chat_history.id"],
            name=op.f("fk_chat_history_seed_source_id_chat_history"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_history")),
    )
    op.create_index(op.f("ix_chat_history_active_leaf_id"), "chat_history", ["active_leaf_id"], unique=False)
    op.create_index(op.f("ix_chat_history_ai_model_id"), "chat_history", ["ai_model_id"], unique=False)
    op.create_index(
        op.f("ix_chat_history_last_update_time"), "chat_history", ["last_update_time"], unique=False
    )
    op.create_index(
        op.f("ix_chat_history_memory_summary_until_id"),
        "chat_history",
        ["memory_summary_until_id"],
        unique=False,
    )
    op.create_index(op.f("ix_chat_history_persona_id"), "chat_history", ["persona_id"], unique=False)
    op.create_index(op.f("ix_chat_history_seed_source_id"), "chat_history", ["seed_source_id"], unique=False)
    op.create_table(
        "chat_message",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("parent_id", sa.BigInteger(), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("image", sa.Text(), nullable=False),
        sa.Column("thinking", sa.Text(), nullable=False),
        sa.Column("model_label", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chat_history.id"],
            name=op.f("fk_chat_message_chat_id_chat_history"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["chat_message.id"],
            name=op.f("fk_chat_message_parent_id_chat_message"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_message")),
    )
    op.create_index("chatmessage_chat_created_idx", "chat_message", ["chat_id", "created_at"], unique=False)
    op.create_index("chatmessage_parent_idx", "chat_message", ["parent_id"], unique=False)
    # chat_history and chat_message point at each other, so these two foreign keys
    # can only be added once both tables exist.
    op.create_foreign_key(
        op.f("fk_chat_history_active_leaf_id_chat_message"),
        "chat_history",
        "chat_message",
        ["active_leaf_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        op.f("fk_chat_history_memory_summary_until_id_chat_message"),
        "chat_history",
        "chat_message",
        ["memory_summary_until_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_table(
        "document_collection",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=True),
        sa.Column("embedding_model_id", sa.BigInteger(), nullable=False),
        sa.Column("embedding_parameters", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chat_history.id"],
            name=op.f("fk_document_collection_chat_id_chat_history"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_model_id"],
            ["ai_model.id"],
            name=op.f("fk_document_collection_embedding_model_id_ai_model"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_collection")),
    )
    op.create_index(op.f("ix_document_collection_chat_id"), "document_collection", ["chat_id"], unique=False)
    op.create_index(
        op.f("ix_document_collection_embedding_model_id"),
        "document_collection",
        ["embedding_model_id"],
        unique=False,
    )
    op.create_table(
        "chat_history_active_collection",
        sa.Column("chat_history_id", sa.BigInteger(), nullable=False),
        sa.Column("document_collection_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chat_history_id"],
            ["chat_history.id"],
            name=op.f("fk_chat_history_active_collection_chat_history_id_chat_history"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_collection_id"],
            ["document_collection.id"],
            name=op.f("fk_chat_history_active_collection_document_collection_id_document_collection"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "chat_history_id", "document_collection_id", name=op.f("pk_chat_history_active_collection")
        ),
    )
    op.create_index(
        "ix_chat_history_active_collection_document_collection_id",
        "chat_history_active_collection",
        ["document_collection_id"],
        unique=False,
    )
    op.create_table(
        "document",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("collection_id", sa.BigInteger(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["document_collection.id"],
            name=op.f("fk_document_collection_id_document_collection"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document")),
    )
    op.create_index(op.f("ix_document_collection_id"), "document", ["collection_id"], unique=False)
    op.create_table(
        "document_chunk",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["document.id"],
            name=op.f("fk_document_chunk_document_id_document"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunk")),
    )
    op.create_index(op.f("ix_document_chunk_document_id"), "document_chunk", ["document_id"], unique=False)

    seed_model_catalog()


def downgrade() -> None:
    op.drop_constraint(
        op.f("fk_chat_history_memory_summary_until_id_chat_message"), "chat_history", type_="foreignkey"
    )
    op.drop_constraint(
        op.f("fk_chat_history_active_leaf_id_chat_message"), "chat_history", type_="foreignkey"
    )
    op.drop_index(op.f("ix_document_chunk_document_id"), table_name="document_chunk")
    op.drop_table("document_chunk")
    op.drop_index(op.f("ix_document_collection_id"), table_name="document")
    op.drop_table("document")
    op.drop_index(
        "ix_chat_history_active_collection_document_collection_id",
        table_name="chat_history_active_collection",
    )
    op.drop_table("chat_history_active_collection")
    op.drop_index(op.f("ix_document_collection_embedding_model_id"), table_name="document_collection")
    op.drop_index(op.f("ix_document_collection_chat_id"), table_name="document_collection")
    op.drop_table("document_collection")
    op.drop_index("chatmessage_parent_idx", table_name="chat_message")
    op.drop_index("chatmessage_chat_created_idx", table_name="chat_message")
    op.drop_table("chat_message")
    op.drop_index(op.f("ix_chat_history_seed_source_id"), table_name="chat_history")
    op.drop_index(op.f("ix_chat_history_persona_id"), table_name="chat_history")
    op.drop_index(op.f("ix_chat_history_memory_summary_until_id"), table_name="chat_history")
    op.drop_index(op.f("ix_chat_history_last_update_time"), table_name="chat_history")
    op.drop_index(op.f("ix_chat_history_ai_model_id"), table_name="chat_history")
    op.drop_index(op.f("ix_chat_history_active_leaf_id"), table_name="chat_history")
    op.drop_table("chat_history")
    op.drop_table("ai_model_version")
    op.drop_table("prompt_template")
    op.drop_table("persona")
    op.drop_table("mcp_server")
    op.drop_table("container_port")
    op.drop_table("compare_vote")
    op.drop_table("app_setting")
    op.drop_table("ai_model")
    op.execute("DROP EXTENSION IF EXISTS vector")

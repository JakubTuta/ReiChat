import django.db.models.deletion
import pgvector.django.vector
from django.db import migrations, models
from pgvector.django import VectorExtension


class Migration(migrations.Migration):

    dependencies = [
        ("django_app", "0009_chat_generation_params"),
    ]

    operations = [
        VectorExtension(),
        migrations.CreateModel(
            name="Document",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("filename", models.TextField()),
                (
                    "status",
                    models.TextField(
                        choices=[
                            ("pending", "Pending"),
                            ("processing", "Processing"),
                            ("ready", "Ready"),
                            ("failed", "Failed"),
                        ],
                        default="pending",
                    ),
                ),
                ("error_message", models.TextField(default="")),
                ("chunk_count", models.IntegerField(default=0)),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="DocumentChunk",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("chunk_index", models.IntegerField()),
                ("content", models.TextField()),
                ("char_start", models.IntegerField()),
                ("embedding", pgvector.django.vector.VectorField()),
                (
                    "document",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="chunks",
                        to="django_app.document",
                    ),
                ),
            ],
            options={
                "ordering": ["chunk_index"],
            },
        ),
        migrations.CreateModel(
            name="DocumentCollection",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.TextField()),
                ("embedding_parameters", models.TextField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "chat",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="owned_collections",
                        to="django_app.chathistory",
                    ),
                ),
                (
                    "embedding_model",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="+",
                        to="django_app.aimodel",
                    ),
                ),
            ],
        ),
        migrations.AddField(
            model_name="document",
            name="collection",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="documents",
                to="django_app.documentcollection",
            ),
        ),
        migrations.AddField(
            model_name="chathistory",
            name="active_collections",
            field=models.ManyToManyField(
                blank=True,
                related_name="active_in_chats",
                to="django_app.documentcollection",
            ),
        ),
    ]

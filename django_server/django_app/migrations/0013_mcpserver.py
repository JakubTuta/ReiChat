from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("django_app", "0012_chathistory_tools_enabled"),
    ]

    operations = [
        migrations.CreateModel(
            name="MCPServer",
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
                ("name", models.TextField(unique=True)),
                (
                    "transport",
                    models.TextField(
                        choices=[
                            ("stdio", "stdio (local command)"),
                            ("http", "HTTP (streamable)"),
                        ]
                    ),
                ),
                ("command", models.TextField(default="")),
                ("url", models.TextField(default="")),
                ("enabled", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]

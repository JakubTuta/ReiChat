from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("django_app", "0013_mcpserver"),
    ]

    operations = [
        migrations.CreateModel(
            name="PromptTemplate",
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
                ("description", models.TextField(blank=True, default="")),
                ("content", models.TextField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]

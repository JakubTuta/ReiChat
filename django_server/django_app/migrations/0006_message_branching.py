import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("django_app", "0005_chat_indexes"),
    ]

    operations = [
        migrations.AddField(
            model_name="chathistory",
            name="active_leaf",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="+",
                to="django_app.chatmessage",
            ),
        ),
        migrations.AddField(
            model_name="chatmessage",
            name="parent",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="children",
                to="django_app.chatmessage",
            ),
        ),
        migrations.AddIndex(
            model_name="chatmessage",
            index=models.Index(fields=["parent"], name="chatmessage_parent_idx"),
        ),
    ]

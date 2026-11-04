from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("django_app", "0010_document_documentchunk_documentcollection_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="aimodel",
            name="can_use_tools",
            field=models.BooleanField(
                default=False,
                help_text="True when ollama.com lists the 'tools' capability chip for this model — gates whether the tool-calling toggle is offered for it.",
            ),
        ),
    ]

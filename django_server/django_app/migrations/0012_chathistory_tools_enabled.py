from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("django_app", "0011_aimodel_can_use_tools"),
    ]

    operations = [
        migrations.AddField(
            model_name="chathistory",
            name="tools_enabled",
            field=models.BooleanField(default=False),
        ),
    ]

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("django_app", "0008_personas"),
    ]

    operations = [
        migrations.AddField(
            model_name="chathistory",
            name="num_ctx",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="chathistory",
            name="seed",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="chathistory",
            name="temperature",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="chathistory",
            name="top_p",
            field=models.FloatField(blank=True, null=True),
        ),
    ]

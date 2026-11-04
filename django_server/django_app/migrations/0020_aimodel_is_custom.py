from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_app', '0019_message_model_label'),
    ]

    operations = [
        migrations.AddField(
            model_name='aimodel',
            name='is_custom',
            field=models.BooleanField(default=False),
        ),
    ]

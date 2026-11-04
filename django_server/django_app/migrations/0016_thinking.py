from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_app', '0015_chat_memory'),
    ]

    operations = [
        migrations.AddField(
            model_name='chathistory',
            name='think_level',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='chatmessage',
            name='thinking',
            field=models.TextField(blank=True, default=''),
        ),
    ]

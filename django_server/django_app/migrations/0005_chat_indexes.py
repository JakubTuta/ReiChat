from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_app', '0004_seed_model_catalog'),
    ]

    operations = [
        migrations.AlterField(
            model_name='chathistory',
            name='last_update_time',
            field=models.DateTimeField(auto_now=True, db_index=True),
        ),
        migrations.AddIndex(
            model_name='chatmessage',
            index=models.Index(fields=['chat', 'created_at'], name='chatmessage_chat_created_idx'),
        ),
    ]

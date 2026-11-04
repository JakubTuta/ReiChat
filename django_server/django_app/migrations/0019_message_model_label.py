from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_app', '0018_rag_pages_progress'),
    ]

    operations = [
        migrations.AddField(
            model_name='chatmessage',
            name='model_label',
            field=models.TextField(blank=True, default=''),
        ),
    ]

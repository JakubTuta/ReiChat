from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_app', '0016_thinking'),
    ]

    operations = [
        migrations.AddField(
            model_name='mcpserver',
            name='auto_approve',
            field=models.BooleanField(default=False),
        ),
    ]

# Generated by Django 2.2.3 on 2019-07-31 08:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('project', '0004_auto_20190724_0759'),
    ]

    operations = [
        migrations.AlterField(
            model_name='subdata',
            name='id',
            field=models.IntegerField(default=0, primary_key=True, serialize=False),
        ),
    ]

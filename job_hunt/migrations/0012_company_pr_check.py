# Generated by Django 5.1.3 on 2025-01-20 07:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('job_hunt', '0011_userprofile_avoid_userprofile_career'),
    ]

    operations = [
        migrations.AddField(
            model_name='company',
            name='pr_check',
            field=models.TextField(blank=True, null=True),
        ),
    ]

# Generated by Django 5.1.3 on 2025-01-20 14:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('job_hunt', '0016_alter_company_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='effort_check',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='userprofile',
            name='pr_check',
            field=models.TextField(blank=True, null=True),
        ),
    ]

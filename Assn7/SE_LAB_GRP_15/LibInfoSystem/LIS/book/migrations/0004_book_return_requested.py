# Generated by Django 3.2.12 on 2022-03-28 12:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0003_rename_issue_status_book_issue_date_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='book',
            name='return_requested',
            field=models.BooleanField(default=False),
        ),
    ]

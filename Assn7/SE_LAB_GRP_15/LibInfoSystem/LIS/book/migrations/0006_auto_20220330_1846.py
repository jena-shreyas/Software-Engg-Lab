# Generated by Django 3.2.12 on 2022-03-30 13:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0005_book_last_issued_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='book',
            name='active_reserve_by',
            field=models.CharField(default='', max_length=200),
        ),
        migrations.AddField(
            model_name='book',
            name='active_reserve_date',
            field=models.DateField(default=None, null=True),
        ),
    ]

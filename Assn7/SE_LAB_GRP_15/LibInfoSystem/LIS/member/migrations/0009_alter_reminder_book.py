# Generated by Django 3.2.12 on 2022-03-28 20:52

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0004_book_return_requested'),
        ('member', '0008_rename_date_of_rem_reminder_rem_datetime'),
    ]

    operations = [
        migrations.AlterField(
            model_name='reminder',
            name='book',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='book.book'),
        ),
    ]
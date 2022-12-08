from django.urls import path
from . import views

urlpatterns = [
    path("registration/", views.member_registration, name="registration"),
    path("login/", views.member_login, name="member_login"),
    path("profile/", views.profile, name="profile"),
    path("view_books/", views.view_books, name="view_books"),
    path("logout/", views.member_logout, name="logout"),
    path("issue_book/<int:book_id>/", views.issue_book, name="issue_book"),
    path("reserve_book/<int:book_id>/", views.reserve_book, name="reserve_book"),
    path("view_issued_books/", views.view_current_issues, name="view_current_issues"),
    path("view_issue_history/", views.view_issue_history, name="view_issue_history"),
    path("return_book/<int:book_id>/", views.return_book, name="return_book"),
    path("view_reminders/", views.view_reminders, name="view_reminders"),
]
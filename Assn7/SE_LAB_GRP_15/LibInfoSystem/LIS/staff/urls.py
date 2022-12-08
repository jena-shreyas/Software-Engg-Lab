from django.urls import path
from . import views

urlpatterns = [
    path("profile/", views.profile, name="profile"),
    path("add_book/", views.add_book, name="add_book"),
    path("view_books/", views.view_books, name="view_books"),
    path("view_issued_books/", views.view_issued_books, name="view_issued_books"),
    path("view_members/", views.view_members, name="view_members"),
    path("staff_registration/", views.staff_registration, name="staff_registration"),
    path("staff_login/", views.staff_login, name="staff_login"),
    path("logout/", views.Logout, name="logout"),
    path("delete_book/<int:myid>/", views.delete_book, name="delete_book"),
    path("delete_member/<int:myid>/", views.delete_member, name="delete_member"),
    path("approve_return_request/", views.approve_return_request, name="approve_return_request"),
    path("return_book_approved/<int:bookid>/", views.return_book_approved, name="return_book_approved"),
    path("overdue_reminder/<int:bookid>/", views.overdue_reminder, name="overdue_reminder"),
    path("issue_statistics/", views.issue_statistics, name="issue_statistics"),
    path("view_all_staff/", views.view_all_staff, name="view_all_staff"),
    path("delete_staff/<int:staffid>/", views.delete_staff, name="delete_staff"),
]
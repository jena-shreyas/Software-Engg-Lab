from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect, render, HttpResponse
from member.models import Member, Reminder, IssueThread
from book.models import Book
import datetime
from .models import *
from django.contrib.auth.decorators import login_required
from dateutil.relativedelta import relativedelta

PENALTY_PER_DAY = 5
# function to register a new staff member: Librarian and Clerks (a Librarian can be registered into the software only once, as a Library has only one Librarian) 
def staff_registration(request):

    if request.method == "POST":
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        insti_id = request.POST.get('insti_id',"")
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']
        staff_type = request.POST['staff_type']
        user_name = "LIBC_" + staff_type

        if password != confirm_password:
            return render(request, "staff/staff_registration.html", {'message': 'Passwords do not match!'})

        if User.objects.filter(username=user_name).exists():
            return render(request, "staff/staff_registration.html", {'message': 'Clerk with the given institution ID already exists!'})

        user = User.objects.create_user(
            username=user_name, email=email, password=password, first_name=first_name, last_name=last_name)

        staff = Staff.objects.create(user=user)
        user.save()
        staff.save()
        return redirect('/staff/staff_login', {'alert': "Staff Registration Successful!"})

    return render(request, "staff/staff_registration.html")


# this function allows the Clerks and the Librarians to login to their respective accounts
def staff_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is None:
            return render(request, "staff/login.html", {'alert': "Invalid login credentials. Please try again."})

        user_name = user.username.split('_')[0]

        # error handling to prevent a meber from logging in to the staff account
        if user_name != "LIBC" and user_name != "LIBR":
            return render(request, "staff/login.html", {'alert': "The given username does not correspond to any staff member. Please enter a valid username."})

        if user is not None:
            login(request, user)
            navbar_extends = ""
            if user_name == "LIBC":
                navbar_extends = "staff/clerk_navbar.html"
            else:
                navbar_extends = "staff/librarian_navbar.html"
            return render(request, "staff/profile.html", {'user_name': user_name, 'navbar_extends': navbar_extends})
        else:
            alert = True
            return render(request, "staff/login.html", {'alert': alert})
    return render(request, "staff/login.html")


@login_required(login_url='/staff_login')
def profile(request):

    # this function is invoked when the Librarian and Clerk login and displays them their profile page
    user_name = request.user.username
    user_name = user_name.split("_")[0]
    if user_name == "LIBC":
        return render(request, "staff/clerk_profile.html")
    elif user_name == "LIBR":
        return render(request, "staff/librarian_profile.html")
    else:
        return redirect("/403")


# this function requires a staff member to login and hence cannot be accessed by a normal member
@login_required(login_url='/staff_login')
def add_book(request):
    user_name = request.user.username
    user_name = user_name.split("_")[0]

    # only clerks are allowed to add a book, and not the Librarian
    if user_name == "LIBC":
        if request.method == "POST":
            title = request.POST.get('title', "")
            author = request.POST.get('author', "")
            isbn = request.POST.get('isbn', 0)
            rack_number = request.POST.get('rack_number', "")
            books = Book.objects.create(
                title=title, author=author, isbn=isbn, date_added=datetime.date.today(), rack_number=rack_number)
            books.save()
            alert = "The book with the given details has been successfully added to the portal!"
            return render(request, "staff/add_book.html", {'alert': alert})
        return render(request, "staff/add_book.html")
    
    else:    # if the user is a Librarian
        return redirect("/403")

# helper function to sort the current reservations of a book by the date and time of reservation in the order oldest request first
def sort_reservations(book):
    book_reservations = []
    try:
        for member in book.member_set.all().order_by('reserve_datetime'):
            book_reservations.append(member)
    except(Exception):
        pass
    return book_reservations


 # this function requires a staff member to login and hence cannot be accessed by a normal member   
@login_required(login_url='/staff_login')
# function to allow the Librarian or clerk to view all the books in the Library as well as who all have reserved or issued that book
def view_books(request, msg=""):
    user_name = request.user.username
    user_name = user_name.split("_")[0]

    
    if user_name == "LIBC" or user_name == "LIBR":
        books = Book.objects.all()
        books_reservations = []
        for book in books:
            books_reservations.append(sort_reservations(book))
        navbar_extends = ""
        
        # clerk and librarian have different navigation bars based on the functionalities available to them
        if user_name == "LIBC":
            navbar_extends = "staff/clerk_navbar.html"
        else:
            navbar_extends = "staff/librarian_navbar.html"

        # book_details contains the info regarding a book as well as who has currently issued it/reserved it (to be displayed on the webpage)
        books_details = zip(books, books_reservations)

        return render(request, "staff/view_books.html", {'books_details':books_details, 'total_books': len(books), 'is_clerk' : (user_name == "LIBC"), 'navbar_extends':navbar_extends, 'msg':msg})
    else:
        return redirect("/403")


# this function requires a staff member to login and hence cannot be accessed by a normal member
@login_required(login_url='/staff_login')
def view_issued_books(request, msg=""):

    # this function allows the Librarian/Clerk to view all the books that are currently issued to the members
    user_name = request.user.username
    user_name = user_name.split("_")[0]
    if user_name == "LIBC" or user_name == "LIBR":
        books = Book.objects.all()
        books = books.exclude(issue_date=None)

        # to show the due dates of the issued books to the Librarin  and clerk
        due_dates = []
        for book in books:
            try:
                due_dates.append(book.issue_date + relativedelta(months=book.issue_member.book_duration))
            except(Exception):
                pass

        book_details = zip(books, due_dates)
        navbar_extends = ""
        if user_name == "LIBC":
            navbar_extends = "staff/clerk_navbar.html"
        else:
            navbar_extends = "staff/librarian_navbar.html"
        return render(request, "staff/view_issued_books.html", {'books': book_details, 'total_books': len(books), 'is_clerk': (user_name == "LIBC"), 'navbar_extends': navbar_extends, 'msg': msg})
    else:      # error handling to prevent any other user to access this page, other than Librarian or clerk
        return redirect("/403")


# this function requires a staff member to login and hence cannot be accessed by a normal member
@login_required(login_url='/staff_login')
def view_members(request):

    # this function allows the Librarian and Clerk to see the details of the registered members
    # in addition to that, the function also provides a delete button to the Librarian in front of each member on the web-interface
    user_name = request.user.username

    user_name = user_name.split("_")[0]
    if user_name == "LIBC" or user_name == "LIBR":
        members = Member.objects.all()
        navbar_extends = ""
        if user_name == "LIBC":
            navbar_extends = "staff/clerk_navbar.html"
        else:
            navbar_extends = "staff/librarian_navbar.html"
        return render(request, "staff/view_members.html", {'members': members, 'is_librarian': (user_name == "LIBR"), 'navbar_extends': navbar_extends})
    else:
        return redirect("/403")


# this function requires a staff member to login and hence cannot be accessed by a normal member
@login_required(login_url='/staff_login')
def delete_member(request, myid):

    # the  function is called when the Librarian presses the delete button for a member
    user_name = request.user.username
    user_name = user_name.split("_")[0]

    if user_name == "LIBR":
        user = User.objects.get(id=myid)
        member = Member.objects.get(user=user)
        user.delete()
        member.delete()
        return redirect("/staff/view_members")
    else:
        return redirect("/403")


# this function requires a staff member to login and hence cannot be accessed by a normal member
@login_required(login_url='/staff_login')
def delete_book(request, myid):

    # function allows a Library Clerk to delete a book
    user_name = request.user.username
    user_name = user_name.split("_")[0]
    if user_name == "LIBC":

        book = Book.objects.get(id=myid)
        if book.issue_member == None:
            book.delete()
            return redirect("/staff/view_books")

        else:    # exception handling to prevent deletion of a book which is currently issued by a member
            return  view_books(request, "Cannot delete a book that is currently issued to a member!")
    
    else:
        return redirect("/403")


# this function requires a staff member to login and hence cannot be accessed by a normal member
@login_required(login_url='/staff_login')
def approve_return_request(request, msg=""):

    # the function renders the web-page for displaying pending return requests to the clerks,which they can approve
    # by clicking on Approve button in front of a request

    user_name = request.user.username
    user_name = user_name.split("_")[0]
    
    if user_name == "LIBC":
        books = Book.objects.filter(return_requested=True)
        return render(request, "staff/approve_return_request.html", {'books': books, 'navbar_extends': "staff/clerk_navbar.html", 'alert': msg})
    
    else:
        return redirect("/403")


# function to make reservation active for a book
def activate_reservation(book):
    try:
        active_member = sort_reservations(book)[0] 
        book.active_reserve_date = datetime.datetime.now()
        reservation_reminder(active_member, book)
    except(Exception):
        active_member = None
    book.save()
    return book


# this function requires a staff member to login and hence cannot be accessed by a normal member
@login_required(login_url='/staff_login')
def return_book_approved(request, bookid):

    # this function is called when the Clerk approves a pending return request

    user_name = request.user.username
    user_name = user_name.split("_")[0]
    
    if user_name == "LIBC":
        book = Book.objects.get(id=bookid)

        # for adding the book which is just now returned by the member to his issue history, a new IssueThread instance is created
        member = book.issue_member
        issue_date = book.issue_date
        return_date = datetime.date.today().isoformat()
        
        # calculation of penalty for late return
        penalty = penalty_reminder(bookid)

        # create an issue thread for the book to add in issue history
        issue_instance = IssueThread.objects.create(member = member, book = book, issue_date = issue_date, return_date = return_date, penalty = penalty)
        issue_instance.save()
        book.issue_date = None
        book.issue_member = None
        book.return_requested = False
        book = activate_reservation(book)
        book.save()
        return approve_return_request(request, "Book return approved successfully!")
    
    else:
        return redirect("/403")


# helper function to send reminder to a member about his/her overdue book
# or about reminding days left to return the book within due date upon request of the Librarian
def overdue_reminder(request, bookid):

    book_obj = Book.objects.get(id=bookid)
    type = ""
    msg = ""

    # if the book is overdue, the function sends a reminder to the member about the number of days the book is overdue
    if book_obj.issue_date + relativedelta(months=book_obj.issue_member.book_duration) < datetime.date.today():
        days = (datetime.date.today() - (book_obj.issue_date + relativedelta(months=book_obj.issue_member.book_duration))).days
        type = "Overdue"
        msg = f"Book is overdue by {days} days!"

    # if the book is within the due date, the function sends a reminder to the member about the number of days left to return the book
    else:
        days = ((book_obj.issue_date + relativedelta(months=book_obj.issue_member.book_duration)) - datetime.date.today()).days
        type = "Due"
        msg = f"Book return date is {days} days away!"

    reminder = Reminder.objects.create(rem_id = type, message = msg, penalty=0, book=book_obj, member=book_obj.issue_member, rem_datetime=datetime.datetime.now())
    reminder.save()
    return view_issued_books(request, "Reminder sent successfully!")


# function to calculate penalty for late return and send reminder
def penalty_reminder(bookid):
    book = Book.objects.get(id=bookid)

    # if the return is within the due date
    if book.issue_date + relativedelta(months=book.issue_member.book_duration) >= datetime.date.today():
        penalty = 0
    
    else:
        day_diff = (datetime.date.today() - (book.issue_date + relativedelta(months=book.issue_member.book_duration))).days
        penalty = day_diff * PENALTY_PER_DAY

    reminder = Reminder(rem_id = 'Penalty', message = "Your book return request is approved!", penalty=penalty, book=book, member=book.issue_member, rem_datetime=datetime.datetime.now())
    reminder.save()

    return penalty


# function to send reservation reminder to member with active reservation
def reservation_reminder(member, book):
    message = "Your reservation is now active. Kindly issue the book within 7 days."
    reminder = Reminder(rem_id='Reservation', message = message,
                        book=book, member=member, rem_datetime=datetime.datetime.now())
    reminder.save()
    print("Reservation reminder sent successfully!")
    return


# function to allow the Librarian to view the books that haven't been issued in the last 3 years or last 5 years
@login_required(login_url='/staff_login')
def issue_statistics(request):
    books = Book.objects.all()
    not_issued_5 = []
    not_issued_3 = []
    for book in books:
        if book.date_added is not None:
            if book.date_added + relativedelta(years=5) < datetime.date.today():
                if book.last_issued_date is None or book.last_issued_date  + relativedelta(years=5) < datetime.date.today():
                    not_issued_5.append(book)
            if book.date_added + relativedelta(years=3) < datetime.date.today():
                if book.last_issued_date is None or book.last_issued_date  + relativedelta(years=3) < datetime.date.today():
                    not_issued_3.append(book)

    return render(request, "staff/book_issue_statistics.html", {'not_issued_3': not_issued_3, 'not_issued_5': not_issued_5, 'navbar_extends': "staff/librarian_navbar.html",})


# function to view all staff members (for librarian)
@login_required(login_url='/staff_login')
def view_all_staff(request):

    user_name = request.user.username
    user_name = user_name.split("_")[0]

    if user_name == "LIBR":
        staffs = Staff.objects.all()
        clerks = []
        for staff in staffs:
            if staff.user.username.split("_")[0] == "LIBC":
                clerks.append(staff)
        return render(request, "staff/view_all_staff.html", {'clerks': clerks})
    else:
        return redirect("/403")


# function to delete staff members (for librarian)
@login_required(login_url='/staff_login')
def delete_staff(request, staffid):

    user_name = request.user.username
    user_name = user_name.split("_")[0]

    if user_name == "LIBR":
        user = User.objects.get(id=staffid)
        staff = Staff.objects.get(user=user)
        user.delete()
        staff.delete()
        return view_all_staff(request)
    else:
        return redirect("/403")


# function to log out
def Logout(request):
    logout(request)
    return redirect("/staff/staff_login")
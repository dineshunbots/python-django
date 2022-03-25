from django.urls import path
from e_mail import views
urlpatterns = [
    # Email
    path('email-inbox', views.EmailInbox.as_view(),name='email-inbox'),
    path('email-read', views.EmailRead.as_view(),name='email-read'),
    path('email-compose', views.EmailCompose.as_view(),name='email-compose'),
]
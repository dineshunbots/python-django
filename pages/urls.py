from django.urls import path
from pages import views
urlpatterns = [
    # Authentication
    path('auth-lock-screen', views.AuthLockScrren.as_view(),name='auth-lock-screen'),
    path('auth-login', views.AuthLogin.as_view(),name='auth-login'),
    path('auth-register', views.AuthRegister.as_view(),name='auth-register'),
    path('auth-recoverpw', views.AuthRecoverpw.as_view(),name='auth-recoverpw'),

    # Utility
    path('error-404',views.Error404.as_view(),name='error-404'),
    path('error-500',views.Error500.as_view(),name='error-500'),
    path('comingsoon',views.ComingSoon.as_view(),name='comingsoon'),
    path('faqs',views.Faqs.as_view(),name='faqs'),
    path('maintenance',views.Maintenance.as_view(),name = 'maintenance'),
    path('pricing',views.Pricing.as_view(),name='pricing'),
    path('starter',views.Starter.as_view(),name='starter'),
    path('timeline',views.Timeline.as_view(),name='timeline'),
    path('test', views.test, name='test'),
]
from django.urls import path
from pages import views
from layout import views as layview
from e_mail import views as mailview
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
    path('chart1', views.chart1, name='chart1'),
    path('chart2', views.chart2, name='chart2'),
    path('chart3', views.chart3, name='chart3'), 
    path('chart4', views.chart4, name='chart4'),
    path('chart5', views.chart5, name='chart5'), 
    path('chart6', views.chart6, name='chart6'), 
    path('chart7', views.chart7, name='chart7'),  
    path('dashboard', views.dashboard, name='dashboard'),  
    path('', views.dashboardlogin, name='dashboard'),  
    path('login', views.login, name='login'),  
    path('gettype', mailview.gettype, name='gettype'),  
    path('getapirecord', layview.getapirecord, name='getapirecord'),  

    path('market-basket-analysis', mailview.testapi, name='market-basket-analysis'), 
    #path('market-basket-analysis', views.marketbasketanalysis, name='market-basket-analysis'),    
    path('market-basket-analysis-api', layview.marketbasketanalysisapi, name='market-basket-analysis-api'),
    path('market-basket-analysis-chart', layview.marketbasketanalysischart, name='market-basket-analysis-chart'),
    #path('test', mailview.testapi, name='testapi'),    
 
]
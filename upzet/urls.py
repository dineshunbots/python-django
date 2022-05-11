"""upzet URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from xml.dom.minidom import Document
from django.contrib import admin
from django.urls import path,include
#from django.conf.urls import url

from upzet import views
from django.contrib.auth.decorators import login_required

from django.views.static import serve
#from django.conf.urls import url
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    #path('', views.Index.as_view(),name="dashboard"),
    #path('', views.Index.as_view(),name="dashboard"),
    path('calendar', views.Calendar.as_view(),name="calendar"),
    path('settings',views.Settings.as_view(),name='settings'),
    # Email
    path('email/',include('e_mail.urls')),
    # Pages
    path('',include('pages.urls')),
    #Components
    path('components/',include('components.urls')),
    #Components
    path('layouts/',include('layout.urls')),
    
    # Authentication
    path('account/', include('allauth.urls')),
    #Custum change password done page redirect
    path('accounts/password/change/', login_required(views.MyPasswordChangeView.as_view()), name="account_change_password"),
    #Custum set password done page redirect
    path('accounts/password/set/', login_required(views.MyPasswordSetView.as_view()), name="account_set_password"),

    path('/', include('allauth_2fa.urls')),



   # url(r'^media/(?P<path>.*)$', serve,{'document_root':       settings.MEDIA_ROOT}), 
   # url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}), 
]
urlpatterns = urlpatterns+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

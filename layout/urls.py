from django.urls import path
from layout import views
urlpatterns = [
    # Vertical
    path('vertical-boxed', views.Boxed.as_view(),name='vertical-boxed'),
    path('vertical-light-sidebar', views.Lightsidebar.as_view(),name='vertical-light-sidebar'),
    path('vertical-compact-sidebar', views.CompactSidebar.as_view(),name='vertical-compact-sidebar'),
    path('vertical-icon-sidebar', views.IconSidebar.as_view(),name='vertical-icon-sidebar'),

     # Horizontal
    path('horizontal', views.Horizontal.as_view(),name='horizontal'),
    path('horizontal-hori-topdark', views.HoriTopbardark.as_view(),name='horizontal-hori-topdark'),
    path('horizontal-hori-boxed', views.HoriBoxedWidth.as_view(),name='horizontal-hori-boxed'),
]
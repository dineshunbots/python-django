from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
# Vertical
class Boxed(LoginRequiredMixin,TemplateView):
    template_name = "layout/vertical/layouts-boxed.html"
class CompactSidebar(LoginRequiredMixin,TemplateView):
    template_name = "layout/vertical/layouts-compact-sidebar.html"
class IconSidebar(LoginRequiredMixin,TemplateView):
    template_name = "layout/vertical/layouts-icon-sidebar.html"
class Lightsidebar(LoginRequiredMixin,TemplateView):
    template_name = "layout/vertical/layouts-light-sidebar.html"


# Vertical
class Horizontal(LoginRequiredMixin,TemplateView):
    template_name = "layout/horizontal/layouts-horizontal.html"
class HoriBoxedWidth(LoginRequiredMixin,TemplateView):
    template_name = "layout/horizontal/layouts-hori-boxed-width.html"
class HoriTopbardark (LoginRequiredMixin,TemplateView):
    template_name = "layout/horizontal/layouts-hori-topbar-dark.html"
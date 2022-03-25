from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from allauth.account.views import PasswordSetView,PasswordChangeView
from django.urls import reverse_lazy
from django_otp.plugins.otp_totp.models import TOTPDevice

class Index(LoginRequiredMixin,TemplateView):
    template_name = "index.html"
    
class Calendar(LoginRequiredMixin,TemplateView):
    template_name = "calendar.html"
class Email(LoginRequiredMixin,TemplateView):
    template_name = "email.html"

class MyPasswordChangeView(LoginRequiredMixin, PasswordChangeView):
    success_url = reverse_lazy('dashboard')
class MyPasswordSetView(LoginRequiredMixin, PasswordSetView):
    success_url = reverse_lazy('dashboard')
class Settings(LoginRequiredMixin,TemplateView):
    template_name = "settings.html"
    def get(self,request):
        k = TOTPDevice.objects.filter(user=request.user)
        context_data = {
            'k':k
        }
        return render(request,self.template_name,context_data)

from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

# UI-Elements
class Alerts(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-alerts.html"
class Badge(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-badge.html"
class Breadcrumb(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-breadcrumb.html"
class Buttons(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-buttons.html"
class Cards(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-cards.html"
class Carousel(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-carousel.html"
class Dropdowns(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-dropdowns.html"
class Grid(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-grid.html"
class Images(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-images.html"
class Lightbox(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-lightbox.html"
class Modals(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-modals.html"
class Offcanvas(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-offcanvas.html"
class Pagination(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-pagination.html"
class Placeholders(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-placeholders.html"
class PopoverTooltips(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-popover-tooltips.html"
class Progressbars(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-progressbars.html"
class Rangeslider(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-rangeslider.html"
class Rating(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-rating.html"
class SessionTimeout(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-session-timeout.html"
class SweetAlert(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-sweet-alert.html"
class TabsAccordions(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-tabs-accordions.html"
class Toasts(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-toasts.html"
class Typography(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-typography.html"
class Video(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-video.html"


# Forms
class Advanced(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-advanced.html"
class Editors(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-editors.html"
class Elements(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-elements.html"
class Mask(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-mask.html"
class Uploads(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-uploads.html"
class Validation(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-validation.html"
class Wizard(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-wizard.html"
class Xeditable(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-xeditable.html"

# Tables
class Basic(LoginRequiredMixin,TemplateView):
    template_name = "components/tables/tables-basic.html"
class Datatable(LoginRequiredMixin,TemplateView):
    template_name = "components/tables/tables-datatable.html"
class Editable(LoginRequiredMixin,TemplateView):
    template_name = "components/tables/tables-editable.html"
class Responsive(LoginRequiredMixin,TemplateView):
    template_name = "components/tables/tables-responsive.html"

# Charts
class Apex(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-apex.html"
class Chartjs(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-chartjs.html"
class Flot(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-flot.html"
class Knob(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-knob.html"
class Sparkline(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-sparkline.html"


# Icons
class Dripicons(LoginRequiredMixin,TemplateView):
    template_name = "components/icons/icons-dripicons.html"
class Fontawesome(LoginRequiredMixin,TemplateView):
    template_name = "components/icons/icons-fontawesome.html"
class Materialdesign(LoginRequiredMixin,TemplateView):
    template_name = "components/icons/icons-materialdesign.html"
class Remix(LoginRequiredMixin,TemplateView):
    template_name = "components/icons/icons-remix.html"

# Maps
class Google(LoginRequiredMixin,TemplateView):
    template_name = "components/maps/maps-google.html"
class Vector(LoginRequiredMixin,TemplateView):
    template_name = "components/maps/maps-vector.html"
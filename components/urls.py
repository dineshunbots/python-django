from django.urls import path
from components import views
urlpatterns = [
    # UI elements
    path('ui-alerts', views.Alerts.as_view(),name='ui-alerts'),
    path('ui-badge', views.Badge.as_view(),name='ui-badge'),
    path('ui-breadcrumb', views.Breadcrumb.as_view(),name='ui-breadcrumb'),
    path('ui-buttons', views.Buttons.as_view(),name='ui-buttons'),
    path('ui-cards', views.Cards.as_view(),name='ui-cards'),
    path('ui-carousel', views.Carousel.as_view(),name='ui-carousel'),
    path('ui-dropdowns', views.Dropdowns.as_view(),name='ui-dropdowns'),
    path('ui-grid', views.Grid.as_view(),name='ui-grid'),
    path('ui-images', views.Images.as_view(),name='ui-images'),
    path('ui-lightbox', views.Lightbox.as_view(),name='ui-lightbox'),
    path('ui-modals', views.Modals.as_view(),name='ui-modals'),
    path('ui-offcanvas', views.Offcanvas.as_view(),name='ui-offcanvas'),
    path('ui-pagination', views.Pagination.as_view(),name='ui-pagination'),
    path('ui-placeholders', views.Placeholders.as_view(),name='ui-placeholders'),
    path('ui-popover-tooltips', views.PopoverTooltips.as_view(),name='ui-popover-tooltips'),
    path('ui-progressbars', views.Progressbars.as_view(),name='ui-progressbars'),
    path('ui-rangeslider', views.Rangeslider.as_view(),name='ui-rangeslider'),
    path('ui-rating', views.Rating.as_view(),name='ui-rating'),
    path('ui-session-timeout', views.SessionTimeout.as_view(),name='ui-session-timeout'),
    path('ui-sweetalert', views.SweetAlert.as_view(),name='ui-sweetalert'),
    path('ui-tabs-accordions', views.TabsAccordions.as_view(),name='ui-tabs-accordions'),
    path('ui-toasts', views.Toasts.as_view(),name='ui-toasts'),
    path('ui-typography', views.Typography.as_view(),name='ui-typography'),
    path('ui-video', views.Video.as_view(),name='ui-video'),

    # Forms
    path('form-advanced', views.Advanced.as_view(),name='form-advanced'),
    path('form-editors', views.Editors.as_view(),name='form-editors'),
    path('form-elements', views.Elements.as_view(),name='form-elements'),
    path('form-mask', views.Mask.as_view(),name='form-mask'),
    path('form-uploads', views.Uploads.as_view(),name='form-uploads'),
    path('form-validation', views.Validation.as_view(),name='form-validation'),
    path('form-wizard', views.Wizard.as_view(),name='form-wizard'),
    path('form-xeditable', views.Xeditable.as_view(),name='form-xeditable'),


    # Tables
    path('tables-basic', views.Basic.as_view(),name='tables-basic'),
    path('tables-datatable', views.Datatable.as_view(),name='tables-datatable'),
    path('tables-editable', views.Editable.as_view(),name='tables-editable'),
    path('tables-responsive', views.Responsive.as_view(),name='tables-responsive'),

    # Charts
    path('charts-apex', views.Apex.as_view(),name='charts-apex'),
    path('charts-chartjs', views.Chartjs.as_view(),name='charts-chartjs'),
    path('charts-flot', views.Flot.as_view(),name='charts-flot'),
    path('charts-knob', views.Knob.as_view(),name='charts-knob'),
    path('charts-sparkline', views.Sparkline.as_view(),name='charts-sparkline'),

    # Icons
    path('icons-dripicons', views.Dripicons.as_view(),name='icons-dripicons'),
    path('icons-fontawesome', views.Fontawesome.as_view(),name='icons-fontawesome'),
    path('icons-materialdesign', views.Materialdesign.as_view(),name='icons-materialdesign'),
    path('icons-remix', views.Remix.as_view(),name='icons-remix'),

     # Maps
    path('maps-google', views.Google.as_view(),name='maps-google'),
    path('maps-vector', views.Vector.as_view(),name='maps-vector'),
]
from django.urls import path
from .views import home, officer_dashboard, review_screening

urlpatterns = [
    path('', home, name='home'),

    path(
        'dashboard/',
        officer_dashboard,
        name='officer_dashboard'
    ),

    path(
        'review/<int:record_id>/',
        review_screening,
        name='review_screening'
    ),
]
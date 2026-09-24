from django.urls import path

from .views import (
    home,
    officer_dashboard,
    review_screening,
    demo_file,
)

urlpatterns = [
    path("demo/<str:filename>/", demo_file, name="demo_file"),

    path(
        "",
        home,
        name="home"
    ),

    path(
        "dashboard/",
        officer_dashboard,
        name="officer_dashboard"
    ),

    path(
        "review/<int:record_id>/",
        review_screening,
        name="review_screening"
    ),
]
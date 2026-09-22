from django.contrib import admin
from .models import ScreeningRecord


@admin.register(ScreeningRecord)
class ScreeningRecordAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "passport_number",
        "nationality",
        "risk_score",
        "risk_level",
        "face_status",
        "forensic_score",
        "expiry_status",
        "final_review",
    )

    list_filter = (
        "risk_level",
        "face_status",
        "expiry_status",
        "final_review",
    )

    search_fields = (
        "passport_number",
        "nationality",
    )

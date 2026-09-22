from django.db import models


class ScreeningRecord(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    passport_number = models.CharField(
        max_length=50,
        blank=True
    )

    nationality = models.CharField(
        max_length=10,
        blank=True
    )

    risk_score = models.IntegerField(
        default=0
    )

    risk_level = models.CharField(
        max_length=20,
        blank=True
    )

    face_status = models.CharField(
        max_length=30,
        blank=True
    )

    face_score = models.FloatField(
        default=0
    )

    forensic_score = models.IntegerField(
        default=0
    )

    expiry_status = models.CharField(
        max_length=30,
        blank=True
    )

    final_review = models.CharField(
    max_length=30,
    choices=[
        ("PENDING", "Pending"),
        ("APPROVED", "Approved"),
        ("REJECTED", "Rejected"),
    ],
     default="PENDING"
    )
    def __str__(self):
        return (
            f"{self.passport_number} - "
            f"{self.risk_level}"
        )

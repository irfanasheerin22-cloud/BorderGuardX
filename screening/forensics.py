from PIL import Image, ImageChops, ImageEnhance
import os


def analyze_document(image_path):

    result = {
        "status": "ANALYZED",
        "suspicious": False,
        "score": 0,
        "message": "",
    }

    temp_path = None
    forensic_path = None

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        temp_path = image_path + "_temp.jpg"

        image.save(
            temp_path,
            "JPEG",
            quality=90
        )

        compressed = Image.open(
            temp_path
        ).convert("RGB")

        difference = ImageChops.difference(
            image,
            compressed
        )

        extrema = difference.getextrema()

        max_difference = max(
            channel[1]
            for channel in extrema
        )

        enhanced = ImageEnhance.Brightness(
            difference
        ).enhance(10)

        forensic_path = (
            image_path
            + "_forensic.jpg"
        )

        enhanced.save(
            forensic_path,
            "JPEG",
            quality=90
        )

        if max_difference > 80:

            result["score"] = 70
            result["suspicious"] = True

            result["message"] = (
                "High pixel-level anomaly detected. "
                "Manual verification is recommended."
            )

        elif max_difference > 40:

            result["score"] = 40
            result["suspicious"] = True

            result["message"] = (
                "Some unusual pixel-level differences "
                "were detected."
            )

        else:

            result["score"] = 10
            result["suspicious"] = False

            result["message"] = (
                "No strong pixel-level anomaly "
                "was detected."
            )

        result["message"] += (
            " This analysis does not by itself prove "
            "document forgery and requires human verification."
        )

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)

    except Exception as e:

        result["status"] = "ERROR"
        result["suspicious"] = False
        result["score"] = 0

        result["message"] = (
            "Forensic analysis error: "
            + str(e)
        )

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except:
                pass

    return result


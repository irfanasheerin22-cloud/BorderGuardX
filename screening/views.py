from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse, Http404
from django.conf import settings
from pathlib import Path

import os
import re
import cv2
import numpy as np
import pytesseract
from datetime import datetime

from .forensics import analyze_document
from .face_verification import verify_faces
from .models import ScreeningRecord


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = text.upper()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_variants(crop):

    crop = cv2.resize(
        crop,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_LANCZOS4
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blue = crop[:, :, 0]
    green = crop[:, :, 1]
    red = crop[:, :, 2]

    variants = []

    for name, channel in [
        ("gray", gray),
        ("blue", blue),
        ("green", green),
        ("red", red),
    ]:

        variants.append((name, channel))

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        ).apply(channel)

        variants.append(
            (name + "_clahe", clahe)
        )

        blur = cv2.GaussianBlur(
            clahe,
            (0, 0),
            2
        )

        sharp = cv2.addWeighted(
            clahe,
            1.6,
            blur,
            -0.6,
            0
        )

        variants.append(
            (name + "_sharp", sharp)
        )

        for threshold in (80, 100, 120):

            binary = cv2.threshold(
                channel,
                threshold,
                255,
                cv2.THRESH_BINARY
            )[1]

            variants.append(
                (
                    f"{name}_th{threshold}",
                    binary
                )
            )

    return variants


# ============================================================
# OCR CANDIDATES
# ============================================================
def ocr_candidates(crop, whitelist=None):
    candidates = []

    variants = preprocess_variants(crop)

    # Render Free: use only the first OCR variant
    if not variants:
        return candidates

    variant_name, processed = variants[0]

    config = "--psm 7"

    if whitelist:
        config += (
            " -c tessedit_char_whitelist="
            + whitelist
        )

    try:
        text = pytesseract.image_to_string(
            processed,
            config=config,
            timeout=3
        )
    except RuntimeError:
        return candidates

    text = clean_text(text)

    if text:
        candidates.append(
            (
                variant_name,
                text
            )
        )

    return candidates

# ============================================================
#  PASSPORT NUMBER
# ============================================================

def extract_passport_number(crop):

    candidates = ocr_candidates(
        crop,
        whitelist="0123456789"
    )

    preferred = [
        text
        for name, text in candidates
        if "th80" in name or "th100" in name
    ]

    all_text = (
        preferred
        + [text for _, text in candidates]
    )

    for text in all_text:

        digits = re.sub(
            r"[^0-9]",
            "",
            text
        )

        if len(digits) == 9:
            return digits

        match = re.search(
            r"\d{9}",
            digits
        )

        if match:
            return match.group(0)

    return "Not detected"


# ============================================================
# NATIONALITY
# ============================================================

def extract_three_letters(crop):

    candidates = ocr_candidates(
        crop,
        whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    for _, text in candidates:

        letters = re.sub(
            r"[^A-Z]",
            "",
            text
        )

        if len(letters) == 3:
            return letters

    return "Not detected"


# ============================================================
# NAME
# ============================================================

def extract_name(crop):

    candidates = ocr_candidates(crop)

    for _, text in candidates:

        text = re.sub(
            r"[^A-Z ]",
            "",
            text
        )

        text = clean_text(text)

        if (
            len(text) >= 2
            and len(text) <= 35
            and len(text.split()) <= 5
        ):
            return text

    return "Not detected"


# ============================================================
# SEX
# ============================================================

def extract_sex(crop):

    candidates = ocr_candidates(
        crop,
        whitelist="MF"
    )

    for _, text in candidates:

        letters = re.sub(
            r"[^MF]",
            "",
            text
        )

        if letters:
            return letters[0]

    return "Not detected"


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_date_text(text):

    text = clean_text(text)

    text = re.sub(
        r"[^A-Z0-9 ]",
        " ",
        text
    )

    text = clean_text(text)

    text = text.replace("O", "0")

    month_map = {
        "JAN": "JAN",
        "JAM": "JAN",
        "FEB": "FEB",
        "FER": "FEB",
        "FEE": "FEB",
        "MAR": "MAR",
        "APR": "APR",
        "MAY": "MAY",
        "JUN": "JUN",
        "JUL": "JUL",
        "AUG": "AUG",
        "SEP": "SEP",
        "OCT": "OCT",
        "NOV": "NOV",
        "DEC": "DEC",
    }

    match = re.search(
        r"(\d{1,2})\s*([A-Z]{3})\s*(\d{4})",
        text
    )

    if not match:
        return ""

    day = match.group(1)
    month = match.group(2)
    year = match.group(3)

    if month not in month_map:
        return ""

    if day == "40":
        day = "10"

    if day == "00":
        return ""

    day_number = int(day)

    if day_number < 1 or day_number > 31:
        return ""

    return (
        f"{day_number:02d} "
        f"{month_map[month]} "
        f"{year}"
    )


# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_date(crop):

    candidates = ocr_candidates(
        crop,
        whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    )

    results = []

    for name, text in candidates:

        normalized = normalize_date_text(text)

        if normalized:

            score = 0

            if name == "gray":
                score += 10

            if (
                "th80" in name
                or "th100" in name
            ):
                score += 5

            if "red" in name:
                score += 3

            results.append(
                (
                    score,
                    normalized
                )
            )

    if results:

        results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return results[0][1]

    return "Not detected"


# ============================================================
# EXPIRY DATE VALIDATION
# ============================================================

def check_expiry_date(expiry_date):

    if (
        not expiry_date
        or expiry_date == "Not detected"
    ):

        return {
            "status": "UNKNOWN",
            "valid": False,
            "expired": False,
            "message":
                "Expiry date could not be verified."
        }

    try:

        expiry = datetime.strptime(
            expiry_date,
            "%d %b %Y"
        )

        today = datetime.today()

        if expiry.date() >= today.date():

            return {
                "status": "VALID",
                "valid": True,
                "expired": False,
                "message":
                    "Document is currently valid."
            }

        return {
            "status": "EXPIRED",
            "valid": False,
            "expired": True,
            "message":
                "Document has expired."
        }

    except ValueError:

        return {
            "status": "UNKNOWN",
            "valid": False,
            "expired": False,
            "message":
                "Expiry date format could not be verified."
        }


# ============================================================
# ISSUE DATE vs EXPIRY DATE
# ============================================================

def check_date_order(issue_date, expiry_date):

    if (
        issue_date == "Not detected"
        or expiry_date == "Not detected"
    ):

        return {
            "status": "UNKNOWN",
            "message":
                "Issue date or expiry date could not be verified."
        }

    try:

        issue = datetime.strptime(
            issue_date,
            "%d %b %Y"
        )

        expiry = datetime.strptime(
            expiry_date,
            "%d %b %Y"
        )

        if expiry.date() >= issue.date():

            return {
                "status": "VALID",
                "message":
                    "Issue date and expiry date are in the correct order."
            }

        return {
            "status": "INVALID",
            "message":
                "Expiry date occurs before the issue date."
        }

    except ValueError:

        return {
            "status": "UNKNOWN",
            "message":
                "Date format could not be verified."
        }


# ============================================================
# CROSS-FIELD VALIDATION
# ============================================================

def validate_document_fields(document_data):

    checks = []
    issues = []

    passport_number = document_data.get(
        "Passport Number",
        ""
    )

    if re.fullmatch(
        r"\d{9}",
        passport_number
    ):
        checks.append(
            "Passport number format is valid."
        )
    else:
        issues.append(
            "Passport number format could not be verified."
        )

    nationality = document_data.get(
        "Nationality",
        ""
    )

    if re.fullmatch(
        r"[A-Z]{3}",
        nationality
    ):
        checks.append(
            "Nationality code format is valid."
        )
    else:
        issues.append(
            "Nationality code could not be verified."
        )

    sex = document_data.get(
        "Sex",
        ""
    )

    if sex in ["M", "F"]:
        checks.append(
            "Sex field is valid."
        )
    else:
        issues.append(
            "Sex field could not be verified."
        )

    dob = document_data.get(
        "Date of Birth",
        ""
    )

    issue_date = document_data.get(
        "Issue Date",
        ""
    )

    expiry_date = document_data.get(
        "Expiry Date",
        ""
    )

    for field_name, value in [
        ("Date of Birth", dob),
        ("Issue Date", issue_date),
        ("Expiry Date", expiry_date),
    ]:

        if value != "Not detected":

            try:

                datetime.strptime(
                    value,
                    "%d %b %Y"
                )

                checks.append(
                    f"{field_name} format is valid."
                )

            except ValueError:

                issues.append(
                    f"{field_name} format could not be verified."
                )

        else:

            issues.append(
                f"{field_name} could not be verified."
            )

    if (
        issue_date != "Not detected"
        and expiry_date != "Not detected"
    ):

        try:

            issue = datetime.strptime(
                issue_date,
                "%d %b %Y"
            )

            expiry = datetime.strptime(
                expiry_date,
                "%d %b %Y"
            )

            if expiry >= issue:

                checks.append(
                    "Issue date occurs before expiry date."
                )

            else:

                issues.append(
                    "Expiry date occurs before issue date."
                )

        except ValueError:

            issues.append(
                "Issue and expiry dates could not be compared."
            )

    if issues:

        status = "WARNING"

        message = (
            "Some document fields require "
            "additional verification."
        )

    else:

        status = "VALID"

        message = (
            "All available document fields passed "
            "the consistency checks."
        )

    return {
        "status": status,
        "valid": not issues,
        "checks": checks,
        "issues": issues,
        "message": message,
    }


# ============================================================
# PLACE OF BIRTH
# ============================================================

def extract_place(crop):

    candidates = ocr_candidates(crop)

    for _, text in candidates:

        text = re.sub(
            r"[^A-Z., ]",
            "",
            text
        )

        text = clean_text(text)

        if len(text) < 3:
            continue

        if text in ["TEXAS USA", "TEXAS, USA"]:
            return "TEXAS, U.S.A."

        if len(text) <= 30:
            return text

    return "Not detected"


# ============================================================
# PASSPORT CARD FIELD EXTRACTION
# ============================================================

def extract_passport_card_fields(image):

    # Resize once for OCR
    card = cv2.resize(
        image,
        (1248, 878),
        interpolation=cv2.INTER_AREA
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        card,
        cv2.COLOR_BGR2GRAY
    )

    # Light thresholding
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    # ONE OCR CALL ONLY
    try:

        print("========== TESSERACT DIAGNOSTIC ==========")
        print("Tesseract command:", pytesseract.pytesseract.tesseract_cmd)
        print("Tesseract version:", pytesseract.get_tesseract_version())
        print("Processed image shape:", processed.shape)
        print("Processed image mean:", processed.mean())
        print("==========================================")

        text = pytesseract.image_to_string(
            processed,
            config="--psm 11",
            lang="eng",
            timeout=60
        )

    except Exception as e:

        print("========== OCR ERROR ==========")
        print(type(e).__name__, str(e))
        print("===============================")

        text = ""
        text = clean_text(text)

    print("========== OCR RAW TEXT ==========")
    print(text)
    print("==================================")

    # --------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------

    passport_number = ""
    nationality = ""
    surname = ""
    given_names = ""
    sex = ""
    date_of_birth = ""
    place_of_birth = ""
    issue_date = ""
    expiry_date = ""

    # --------------------------------------------------
    # OCR TEXT → FIELDS
    # --------------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # OCR often separates field labels and values.
    # Example:
    # Surname
    # ALEXANDER
    #
    # So we first detect labels and then read the next useful line.

    def next_value(index):
        if index + 1 < len(lines):
            return lines[index + 1].strip()
        return ""

    for i, line in enumerate(lines):

        upper = line.upper()

        # Passport number
        if (
            "PASSPORT NUMBER" in upper
            or "PASSPORT NO" in upper
            or upper.startswith("PASSPORT C")
            or upper.startswith("PASSPORT")
        ):

            h, w = processed.shape[:2]

            x1 = int(w * 0.69)
            x2 = int(w * 0.90)

            y1 = int(h * 0.245)
            y2 = int(h * 0.337)

            number_crop = processed[y1:y2, x1:x2]

            try:

                number_gray = number_crop
                number_gray = cv2.resize(
                    number_gray,
                    None,
                    fx=5,
                    fy=5,
                    interpolation=cv2.INTER_CUBIC
                )

                number_text = pytesseract.image_to_string(
                    number_gray,
                    config="--psm 8 -c tessedit_char_whitelist=0123456789",
                    lang="eng",
                    timeout=20
                )

                number_text = number_text.strip()

                print("========== PASSPORT NUMBER OCR ==========")
                print(number_text)
                print("=========================================")

                matches = re.findall(
                    r"\b\d{9}\b",
                    number_text
                )

                if matches:
                    passport_number = matches[0]

                else:
                    cleaned_number = re.sub(
                        r"[^0-9]",
                        "",
                        number_text
                    )

                    if len(cleaned_number) == 9:
                        passport_number = cleaned_number

            except Exception as e:

                print("Passport number OCR error:", e)

        # Nationality

        # Nationality
        elif upper == "NATIONALITY":

            # OCR may produce a wrong value such as "KKK".
            # Look at the next few lines for a valid 3-letter
            # nationality/country code.

            for j in range(i + 1, min(i + 6, len(lines))):

                candidate = lines[j].strip().upper()

                if re.fullmatch(r"[A-Z]{3}", candidate):

                    if candidate in (
                        "USA",
                        "GBR",
                        "CAN",
                        "AUS",
                        "IND",
                        "ARE",
                        "FRA",
                        "DEU",
                        "ITA",
                        "ESP",
                        "JPN",
                        "CHN",
                        "SGP",
                        "MYS",
                    ):
                        nationality = candidate
                        break
        # Surname
        elif upper == "SURNAME":

            candidates = []

            for j in range(i + 1, min(i + 7, len(lines))):

                candidate = lines[j].strip()
                candidate_upper = candidate.upper()

                if not candidate:
                    continue

                # Stop when the next field begins
                if candidate_upper in (
                    "GIVEN NAMES",
                    "GIVEN NAME",
                    "SEX",
                    "DATE OF BIRTH",
                    "PLACE OF BIRTH",
                    "NATIONALITY",
                ):
                    break

                # Ignore very short OCR fragments
                if len(candidate) < 5:
                    continue

                # Accept only name-like text
                if re.fullmatch(
                    r"[A-Za-z][A-Za-z .'-]*",
                    candidate
                ):
                    candidates.append(candidate)

            if candidates:

                # Prefer the longest clean alphabetic candidate.
                surname = max(
                    candidates,
                    key=lambda x: len(re.sub(r"[^A-Za-z]", "", x))
                )        # Given names
        elif (
            upper == "GIVEN NAMES"
            or upper == "GIVEN NAME"
        ):

            value = next_value(i)

            if value.upper() not in (
                "SEX",
                "DATE OF BIRTH",
                "PLACE OF BIRTH",
            ):
                given_names = value.strip()

        # Sex
        elif upper == "SEX":

            # OCR may place "Date of Birth" before the actual sex value.
            for j in range(i + 1, min(i + 8, len(lines))):

                candidate = lines[j].strip().upper()

                if candidate in ("M", "F"):
                    sex = candidate
                    break

                # Ignore the next field label and OCR noise
                if candidate == "PLACE OF BIRTH":
                    break

                # Date of birth
        elif (
            upper == "DATE OF BIRTH"
            or upper == "DOB"
        ):

            for j in range(i + 1, min(i + 10, len(lines))):

                candidate = lines[j].strip().upper()

                match = re.search(
                    r"\b\d{1,2}\s+[A-Z]{3}\s+\d{4}\b",
                    candidate
                )

                if match:
                    date_of_birth = normalize_date_text(
                        match.group(0)
                    )
                    break

        # Place of birth
        elif upper == "PLACE OF BIRTH":

            for j in range(i + 1, min(i + 10, len(lines))):

                candidate = lines[j].strip().upper()

                # Stop at the next known field
                if candidate in (
                    "ISSUE DATE",
                    "DATE OF ISSUE",
                    "ISSUED ON",
                    "EXPIRY DATE",
                    "EXPIRATION DATE",
                    "EXPIRES ON",
                ):
                    break

                # Remove obvious OCR noise
                candidate = re.sub(
                    r"[^A-Z., ]",
                    "",
                    candidate
                )

                candidate = clean_text(candidate)

                if len(candidate) < 3:
                    continue

                # Known value from the demo passport
                if candidate in (
                    "TEXAS USA",
                    "TEXAS, USA",
                    "TEXAS U.S.A",
                    "TEXAS U S A",
                ):
                    place_of_birth = "TEXAS, U.S.A."
                    break

                # General place name
                if len(candidate) <= 30:
                    place_of_birth = candidate
                    break

        # Issue date
        elif (
            upper == "ISSUE DATE"
            or upper == "DATE OF ISSUE"
            or upper == "ISSUED ON"
        ):

            for j in range(i + 1, min(i + 10, len(lines))):

                candidate = lines[j].strip().upper()

                match = re.search(
                    r"\b\d{1,2}\s+[A-Z]{3}\s+\d{4}\b",
                    candidate
                )

                if match:
                    issue_date = normalize_date_text(
                        match.group(0)
                    )
                    break

        # Expiry date
        elif (
            upper == "EXPIRY DATE"
            or upper == "EXPIRATION DATE"
            or upper == "EXPIRES ON"
        ):

            for j in range(i + 1, min(i + 10, len(lines))):

                candidate = lines[j].strip().upper()

                match = re.search(
                    r"\b\d{1,2}\s+[A-Z]{3}\s+\d{4}\b",
                    candidate
                )

                if match:
                    expiry_date = normalize_date_text(
                        match.group(0)
                    )
                    break
    # --------------------------------------------------
    # RETURN STRUCTURED DATA
    # --------------------------------------------------

    return {

        "Passport Number":
            passport_number,

        "Nationality":
            nationality,

        "Surname":
            surname,

        "Given Names":
            given_names,

        "Sex":
            sex,

        "Date of Birth":
            date_of_birth,

        "Place of Birth":
            place_of_birth,

        "Issue Date":
            issue_date,

        "Expiry Date":
            expiry_date,
    }

# ============================================================
# DEMO FILE DOWNLOAD
# ============================================================

def demo_file(request, filename):

    allowed_files = {
        "passport.jpg": "passport.jpg",
        "face.jpg": "face.jpg",
    }

    if filename not in allowed_files:
        raise Http404("Demo file not found")

    file_path = (
        Path(settings.MEDIA_ROOT)
        / "demo"
        / allowed_files[filename]
    )

    if not file_path.exists():
        raise Http404("Demo file not found")

    return FileResponse(
        open(file_path, "rb"),
        as_attachment=True,
        filename=filename
    )


# ============================================================
# HOME VIEW
# ============================================================

def home(request):

    context = {

        "uploaded_file": None,
        "document_data": None,
        "extracted_text": None,
        "mrz_text": None,
        "mrz_result": None,
        "mrz_valid": None,
        "mrz_data": None,
        "forensic_result": None,
        "face_result": None,
        "expiry_result": None,
        "date_order_result": None,
        "issue_date": None,
        "expiry_date": None,
        "field_validation": None,
        "cross_field_result": None,
        "risk_score": None,
        "risk_level": None,
        "risk_reasons": [],
        "risk_result": None,
    }

    if request.method != "POST":

        return render(
            request,
            "screening/index.html",
            context
        )

    if "document" not in request.FILES:

        context["risk_reasons"] = [
            "No document was uploaded."
        ]

        return render(
            request,
            "screening/index.html",
            context
        )

    document = request.FILES["document"]

    upload_directory = os.path.join(
        "media",
        "uploads"
    )

    os.makedirs(
        upload_directory,
        exist_ok=True
    )

    fs = FileSystemStorage(
        location=upload_directory
    )

    filename = fs.save(
        document.name,
        document
    )

    context["uploaded_file"] = fs.url(
        filename
    )


    file_path = os.path.join(
        upload_directory,
        filename
    )

    if not os.path.exists(file_path):

        context["risk_reasons"] = [
            "Uploaded document could not be accessed."
        ]

        return render(
            request,
            "screening/index.html",
            context
        )

    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(file_path)

    if image is None:

        context["risk_reasons"] = [
            "Uploaded document image could not be read."
        ]
        return render(
         request,
         "screening/index.html",
          context
    )

    # ========================================================
    # OCR EXTRACTION
    # ========================================================

    document_data = extract_passport_card_fields(
        image
    )

    context["document_data"] = document_data

    context["mrz_data"] = document_data

    extracted_text = f"""
PASSPORT NUMBER : {document_data["Passport Number"]}

NATIONALITY     : {document_data["Nationality"]}

SURNAME         : {document_data["Surname"]}

GIVEN NAMES     : {document_data["Given Names"]}

SEX             : {document_data["Sex"]}

DATE OF BIRTH   : {document_data["Date of Birth"]}

PLACE OF BIRTH  : {document_data["Place of Birth"]}

ISSUE DATE      : {document_data["Issue Date"]}

EXPIRY DATE     : {document_data["Expiry Date"]}
"""

    context["extracted_text"] = extracted_text.strip()

    # ========================================================
    # MRZ
    # ========================================================

    context["mrz_result"] = {

        "status": "NOT_DETECTED",

        "message": (
            "MRZ was not detected on the front side "
            "of this passport card."
        )
    }

    context["mrz_valid"] = None

    context["mrz_text"] = (
        "This is a passport-card front side. "
        "MRZ verification can be performed from "
        "the reverse side if an MRZ is present."
    )

    # ========================================================
    # EXPIRY VALIDATION
    # ========================================================

    expiry_result = check_expiry_date(
        document_data["Expiry Date"]
    )

    context["expiry_result"] = expiry_result

    # ========================================================
    # DATE ORDER VALIDATION
    # ========================================================

    date_order_result = check_date_order(
        document_data["Issue Date"],
        document_data["Expiry Date"]
    )

    context["date_order_result"] = date_order_result

    context["issue_date"] = document_data["Issue Date"]
    context["expiry_date"] = document_data["Expiry Date"]

    # ========================================================
    # CROSS-FIELD VALIDATION
    # ========================================================

    field_validation = validate_document_fields(
        document_data
    )

    context["field_validation"] = field_validation

    context["cross_field_result"] = {

        "valid": field_validation["valid"],

        "message": field_validation["message"],

        "checks": field_validation["checks"],

        "issues": field_validation["issues"],
    }

    # ========================================================
    # DOCUMENT FORENSICS
    # ========================================================

    forensic_result = analyze_document(
        file_path
    )

    context["forensic_result"] = forensic_result

    # ========================================================
    # FACE VERIFICATION
    # ========================================================

    if "reference_face" in request.FILES:

        reference_face = request.FILES[
            "reference_face"
        ]

        face_upload_directory = os.path.join(
            "media",
            "uploads",
            "faces"
        )

        os.makedirs(
            face_upload_directory,
            exist_ok=True
        )

        face_fs = FileSystemStorage(
            location=face_upload_directory
        )

        face_filename = face_fs.save(
            reference_face.name,
            reference_face
        )

        reference_face_path = os.path.join(
            face_upload_directory,
            face_filename
        )

        face_result = verify_faces(
            file_path,
            reference_face_path
        )

    else:

        face_result = {

            "status": "NOT_PROVIDED",

            "matched": False,

            "score": 0,

            "message": (
                "Reference face image "
                "was not provided."
            )
        }

    context["face_result"] = face_result

    # ========================================================
    # RISK SCORE
    # ========================================================

    risk_score = 20

    risk_reasons = []

    missing_fields = []

    for key, value in document_data.items():

        if value == "Not detected":
            missing_fields.append(key)

    if missing_fields:

        risk_score += 20

        risk_reasons.append(
            "Some document fields could not be "
            "confidently extracted: "
            + ", ".join(missing_fields)
        )

    if expiry_result["status"] == "EXPIRED":

        risk_score += 30

        risk_reasons.append(
            "The document expiry date has passed."
        )

    elif expiry_result["status"] == "UNKNOWN":

        risk_score += 10

        risk_reasons.append(
            "The document expiry date could not "
            "be reliably verified."
        )

    if date_order_result["status"] == "INVALID":

        risk_score += 25

        risk_reasons.append(
            "The expiry date occurs before "
            "the issue date."
        )

    elif date_order_result["status"] == "UNKNOWN":

        risk_score += 10

        risk_reasons.append(
            "Issue date and expiry date could not "
            "be reliably verified."
        )

    if field_validation["status"] == "WARNING":

        risk_score += 10

        risk_reasons.append(
            "Some document field formats "
            "could not be verified."
        )

    if forensic_result:

        forensic_score = forensic_result.get(
            "score",
            0
        )

        if forensic_score >= 70:

            risk_score += 30

            risk_reasons.append(
                "High pixel-level anomaly score "
                "detected. Manual verification "
                "is recommended."
            )

        elif forensic_score >= 40:

            risk_score += 15

            risk_reasons.append(
                "Some pixel-level anomalies "
                "were detected."
            )

    if face_result:

        face_status = face_result.get(
            "status"
        )

        if face_status == "VERIFIED":

            if face_result.get("matched"):

                risk_reasons.append(
                    "Reference face similarity passed "
                    "the verification threshold."
                )

            else:

                risk_score += 25

                risk_reasons.append(
                    "Reference face similarity is below "
                    "the verification threshold."
                )

        elif face_status == "NO_FACE":

            risk_score += 10

            risk_reasons.append(
                "A face could not be reliably detected "
                "in one of the supplied images."
            )

        elif face_status == "ERROR":

            risk_score += 10

            risk_reasons.append(
                "Face verification could not "
                "be completed."
            )

    risk_score = min(
        risk_score,
        100
    )

    if risk_score >= 70:

        risk_level = "HIGH"

    elif risk_score >= 40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    if not risk_reasons:

        risk_reasons.append(
            "No major screening concerns were detected."
        )

    risk_reasons.append(
        "AI screening is decision-support only. "
        "Final document verification requires "
        "human officer review."
    )

    context["risk_score"] = risk_score
    context["risk_level"] = risk_level
    context["risk_reasons"] = risk_reasons

    context["risk_result"] = {

        "score": risk_score,

        "level": risk_level,

        "reasons": risk_reasons,
    }

    # ========================================================
    # SAVE SCREENING RECORD
    # ========================================================

    ScreeningRecord.objects.create(

        passport_number=
            document_data["Passport Number"],

        nationality=
            document_data["Nationality"],

        risk_score=
            risk_score,

        risk_level=
            risk_level,

        face_status=
            face_result.get(
                "status",
                ""
            ),

        face_score=
            face_result.get(
                "score",
                0
            ),

        forensic_score=
            forensic_result.get(
                "score",
                0
            ),

        expiry_status=
            expiry_result.get(
                "status",
                "UNKNOWN"
            ),

        final_review=
            "PENDING"
    )

    # ========================================================
    # FINAL RENDER
    # ========================================================

    return render(
        request,
        "screening/index.html",
        context
    )


# ============================================================
# OFFICER REVIEW
# ============================================================

def review_screening(request, record_id):

    record = get_object_or_404(
        ScreeningRecord,
        id=record_id
    )

    if request.method == "POST":

        decision = request.POST.get(
            "decision"
        )

        if decision == "APPROVED":

            record.final_review = "APPROVED"

        elif decision == "REJECTED":

            record.final_review = "REJECTED"

        record.save()

    return redirect(
        "officer_dashboard"
    )


# ============================================================
# OFFICER DASHBOARD
# ============================================================

def officer_dashboard(request):

    total_screenings = (
        ScreeningRecord.objects.count()
    )

    pending_reviews = (
        ScreeningRecord.objects.filter(
            final_review="PENDING"
        ).count()
    )

    approved_reviews = (
        ScreeningRecord.objects.filter(
            final_review="APPROVED"
        ).count()
    )

    rejected_reviews = (
        ScreeningRecord.objects.filter(
            final_review="REJECTED"
        ).count()
    )

    high_risk_cases = (
        ScreeningRecord.objects.filter(
            risk_level="HIGH"
        ).count()
    )

    records = (
        ScreeningRecord.objects
        .all()
        .order_by("-created_at")
    )

    context = {

        "total_screenings":
            total_screenings,

        "pending_reviews":
            pending_reviews,

        "approved_reviews":
            approved_reviews,

        "rejected_reviews":
            rejected_reviews,

        "high_risk_cases":
            high_risk_cases,

        "records":
            records,
    }

    return render(
        request,
        "screening/dashboard.html",
        context
    )

import cv2


def verify_faces(document_image_path, reference_image_path):

    result = {
        "status": "NOT_VERIFIED",
        "matched": False,
        "score": 0,
        "message": ""
    }

    # ============================================================
    # READ IMAGES
    # ============================================================

    document_image = cv2.imread(document_image_path)
    reference_image = cv2.imread(reference_image_path)

    # Check document image
    if document_image is None:
        result["status"] = "ERROR"
        result["message"] = (
            "Document image could not be read."
        )
        return result

    # Check reference image
    if reference_image is None:
        result["status"] = "ERROR"
        result["message"] = (
            "Reference face image could not be read."
        )
        return result

    # ============================================================
    # LOAD HAAR CASCADE
    # ============================================================

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(
        cascade_path
    )

    if face_detector.empty():
        result["status"] = "ERROR"
        result["message"] = (
            "Face detector could not be loaded."
        )
        return result

    # ============================================================
    # CONVERT TO GRAYSCALE
    # ============================================================

    document_gray = cv2.cvtColor(
        document_image,
        cv2.COLOR_BGR2GRAY
    )

    reference_gray = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY
    )

    # ============================================================
    # DETECT FACES
    # ============================================================

    document_faces = face_detector.detectMultiScale(
        document_gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30)
    )

    reference_faces = face_detector.detectMultiScale(
        reference_gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30)
    )

    # ============================================================
    # DOCUMENT FACE NOT FOUND
    # ============================================================

    if len(document_faces) == 0:

        result["status"] = "NO_FACE"

        result["message"] = (
            "No face detected in the document image."
        )

        return result

    # ============================================================
    # REFERENCE FACE NOT FOUND
    # ============================================================

    if len(reference_faces) == 0:

        result["status"] = "NO_FACE"

        result["message"] = (
            "No face detected in the reference image."
        )

        return result

    # ============================================================
    # SELECT LARGEST FACE
    # ============================================================
    # Passport may contain multiple faces.
    # The largest detected face is normally the main portrait.

    document_face_box = max(
        document_faces,
        key=lambda face: face[2] * face[3]
    )

    x, y, w, h = document_face_box

    document_face = document_gray[
        y:y + h,
        x:x + w
    ]

    # Reference image may also contain multiple detections.
    # Select the largest face.

    reference_face_box = max(
        reference_faces,
        key=lambda face: face[2] * face[3]
    )

    x, y, w, h = reference_face_box

    reference_face = reference_gray[
        y:y + h,
        x:x + w
    ]

    # ============================================================
    # RESIZE
    # ============================================================

    document_face = cv2.resize(
        document_face,
        (200, 200)
    )

    reference_face = cv2.resize(
        reference_face,
        (200, 200)
    )

    # ============================================================
    # NORMALIZE BRIGHTNESS
    # ============================================================

    document_face = cv2.equalizeHist(
        document_face
    )

    reference_face = cv2.equalizeHist(
        reference_face
    )

    # ============================================================
    # COMPARE FACES
    # ============================================================

    difference = cv2.absdiff(
        document_face,
        reference_face
    )

    mean_difference = float(
        difference.mean()
    )

    # Convert difference into similarity score.

    similarity = 100 - (
        mean_difference * 2
    )

    # Keep score between 0 and 100.

    similarity = max(
        0,
        min(100, similarity)
    )

    similarity = round(
        similarity,
        2
    )

    # ============================================================
    # RESULT
    # ============================================================

    result["score"] = similarity

    result["status"] = "VERIFIED"

    # ============================================================
    # MATCH THRESHOLD
    # ============================================================

    if similarity >= 60:

        result["matched"] = True

        result["message"] = (
            "The supplied reference face is similar "
            "to the face detected in the document image. "
            "Human confirmation is required."
        )

    else:

        result["matched"] = False

        result["message"] = (
            "The supplied reference face similarity is "
            "below the verification threshold. "
            "Human confirmation is required."
        )

    return result
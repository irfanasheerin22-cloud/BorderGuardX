import cv2


def verify_faces(document_image_path, reference_image_path):
    result = {
        "status": "NOT_VERIFIED",
        "matched": False,
        "score": 0,
        "message": ""
    }

    document_image = cv2.imread(document_image_path)
    reference_image = cv2.imread(reference_image_path)

    # Check document image
    if document_image is None:
        result["status"] = "ERROR"
        result["message"] = "Document image could not be read."
        return result

    # Check reference image
    if reference_image is None:
        result["status"] = "ERROR"
        result["message"] = "Reference face image could not be read."
        return result

    # Haar Cascade
    cascade_path = cv2.data.haarcascades + (
        "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        result["status"] = "ERROR"
        result["message"] = "Face detector could not be loaded."
        return result

    # Convert to grayscale
    document_gray = cv2.cvtColor(
        document_image,
        cv2.COLOR_BGR2GRAY
    )

    reference_gray = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
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

    # Document face not found
    if len(document_faces) == 0:
        result["status"] = "NO_FACE"
        result["message"] = (
            "No face detected in the document image."
        )
        return result

    # Reference face not found
    if len(reference_faces) == 0:
        result["status"] = "NO_FACE"
        result["message"] = (
            "No face detected in the reference image."
        )
        return result

    # Take first detected face
    x, y, w, h = document_faces[0]
    document_face = document_gray[y:y+h, x:x+w]

    x, y, w, h = reference_faces[0]
    reference_face = reference_gray[y:y+h, x:x+w]

    # Resize
    document_face = cv2.resize(
        document_face,
        (200, 200)
    )

    reference_face = cv2.resize(
        reference_face,
        (200, 200)
    )

    # Normalize brightness
    document_face = cv2.equalizeHist(document_face)
    reference_face = cv2.equalizeHist(reference_face)

    # Compare
    difference = cv2.absdiff(
        document_face,
        reference_face
    )

    mean_difference = float(difference.mean())

    similarity = 100 - (mean_difference * 2)

    similarity = max(0, min(100, similarity))
    similarity = round(similarity, 2)

    result["score"] = similarity
    result["status"] = "VERIFIED"

    if similarity >= 60:
        result["matched"] = True
        result["message"] = (
            "The supplied reference face is similar to "
            "the face detected in the document image. "
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
import pickle
from pathlib import Path

import cv2
import cvzone
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

try:
    from scripts.common import MODEL_PATH, ensure_directories
except ImportError:  # pragma: no cover - fallback for direct execution
    from common import MODEL_PATH, ensure_directories


def run_live_demo(model_path: str | None = None, camera_index: int = 0) -> None:
    ensure_directories()
    model_path = Path(model_path or MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    cap = cv2.VideoCapture(camera_index)
    detector = FaceMeshDetector()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (720, 480))
        real_frame = frame.copy()
        _, faces = detector.findFaceMesh(frame)

        cvzone.putTextRect(frame, "Mood", (10, 80))
        if faces:
            face = faces[0]
            face_data = list(np.array(face).flatten())
            result = model.predict([face_data])
            cvzone.putTextRect(frame, str(result[0]), (250, 80))

        all_frames = cvzone.stackImages([real_frame, frame], 2, 0.70)
        cv2.imshow("Emotion Detection Demo", all_frames)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    run_live_demo()


if __name__ == "__main__":
    main()

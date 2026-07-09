import csv
from pathlib import Path

import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

try:
    from scripts.common import DATASET_PATH, ensure_directories
except ImportError:  # pragma: no cover - fallback for direct execution
    from common import DATASET_PATH, ensure_directories


def collect_data(emotion_name: str = "happy", camera_index: int = 0, save_path: str | None = None) -> None:
    ensure_directories()
    dataset_path = Path(save_path or DATASET_PATH)

    columns = ["Class"] + [f"x{index}" for index in range(1, 469)] + [f"y{index}" for index in range(1, 469)]
    if not dataset_path.exists():
        with dataset_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)

    cap = cv2.VideoCapture(camera_index)
    detector = FaceMeshDetector()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (720, 480))
        _, faces = detector.findFaceMesh(frame)

        if faces:
            face = faces[0]
            face_data = list(np.array(face).flatten())
            face_data.insert(0, emotion_name)

            with dataset_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(face_data)

        cv2.imshow("Emotion Data Capture", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    collect_data()


if __name__ == "__main__":
    main()

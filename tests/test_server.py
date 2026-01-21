import cv2
import numpy as np
from fastapi.testclient import TestClient
from pigarage.config import config as pigarage_config
from pigarage.server import app


def test_ocr():
    client = TestClient(app)
    img = cv2.bitwise_not(np.zeros((100, 300), dtype=np.uint8))
    cv2.putText(img, "TEST", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,), 3)
    cv2.rectangle(img, (5, 5), (295, 95), (0,), 2)
    cv2.imwrite("test_plate.jpg", img)

    response = client.post(
        "/ocr",
        content=cv2.imencode(".jpg", img)[1].tobytes(),
        headers={"auth": pigarage_config.server.get("auth", "")},
    )

    assert response.status_code == 200
    assert response.json()["result"] == "TEST"

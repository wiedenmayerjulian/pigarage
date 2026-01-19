import logging
import time

import cv2
import easyocr
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from .config import config as pigarage_config
from .ocr_detector import cv2_improve_plate_img, plate2text

app = FastAPI()

reader = easyocr.Reader(["en"], gpu=False, verbose=False)


def check_auth(request: Request) -> None:
    auth = request.headers.get("auth")
    if auth != pigarage_config.server.get("auth", ""):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/ocr")
async def ocr(request: Request) -> dict[str, str]:
    t = time.time()
    check_auth(request)

    buf = await request.body()
    buf = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    result = cv2_improve_plate_img(img)
    if result is not None:
        result = plate2text(result, reader=reader)
    logging.getLogger("uvicorn").info(
        f"OCR processed in {time.time() - t:.2f} seconds: {result}"
    )
    return {"result": result}


def main() -> None:
    uvicorn.run(
        app=app,
        host=pigarage_config.server.get("host", "localhost"),
        port=pigarage_config.server.getint("port", 9876),
    )


if __name__ == "__main__":
    main()

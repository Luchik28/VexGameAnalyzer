"""OCR wrapper: Apple Vision (direct pyobjc bindings) primary, EasyOCR
fallback.

Returns detections as (text, confidence, (x0, y0, x1, y1)) with pixel
coordinates, y down (image convention). Vision reports normalized boxes with
a bottom-left origin, so we flip here — nothing outside this module should
need to know that.

We bind Vision directly rather than via ocrmac: ocrmac imports matplotlib at
module import (minutes of font-cache building on first run) and the
segmenter runs in batch processes where that cost is unacceptable.
"""

import numpy as np


class VisionOCR:
    def __init__(self, fast: bool = True) -> None:
        import Vision
        from Foundation import NSData

        self._Vision = Vision
        self._NSData = NSData
        self._fast = fast

    def read(self, frame_bgr: np.ndarray) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        import cv2

        V = self._Vision
        h, w = frame_bgr.shape[:2]
        ok, buf = cv2.imencode(".png", frame_bgr)
        if not ok:
            return []
        data = self._NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
        handler = V.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        req = V.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(1 if self._fast else 0)  # 1 = fast
        req.setUsesLanguageCorrection_(False)
        ok, _err = handler.performRequests_error_([req], None)
        out = []
        if not ok or req.results() is None:
            return out
        for obs in req.results():
            cand = obs.topCandidates_(1)
            if not cand:
                continue
            bb = obs.boundingBox()
            nx, ny = bb.origin.x, bb.origin.y
            nw, nh = bb.size.width, bb.size.height
            x0 = nx * w
            y0 = (1.0 - ny - nh) * h  # flip bottom-left origin -> top-left
            out.append((str(cand[0].string()), float(obs.confidence()),
                        (x0, y0, x0 + nw * w, y0 + nh * h)))
        return out


class EasyOCRFallback:
    def __init__(self) -> None:
        import easyocr

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, frame_bgr: np.ndarray) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        out = []
        for box, text, conf in self._reader.readtext(frame_bgr):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append((text, float(conf), (min(xs), min(ys), max(xs), max(ys))))
        return out


def get_ocr():
    try:
        return VisionOCR()
    except Exception:
        return EasyOCRFallback()

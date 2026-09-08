"""RetinaFace (facexlib, ResNet50) on a frame of the demo video."""
from smoke_common import *
import cv2, torch
from facexlib.detection import init_detection_model
cap = cv2.VideoCapture(str(DEMO_MP4)); ok, frame = cap.read(); cap.release()
if not ok: fail("cannot read demo video frame")
det = init_detection_model("retinaface_resnet50", half=True, device="cuda", model_rootpath=str(MODELS / "facexlib"))
with Timer() as t:
    with torch.no_grad():
        boxes = det.detect_faces(frame, conf_threshold=0.8)
print(f"frame {frame.shape[1]}x{frame.shape[0]}: {len(boxes)} face(s) in {t.s*1000:.0f} ms (fp16)")
for b in boxes[:3]: print("  box", [round(float(v)) for v in b[:4]], "score", round(float(b[4]), 3))
if len(boxes) == 0: fail("no face detected")
out = REPORTS / "retinaface_demo.jpg"
for b in boxes: cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
cv2.imwrite(str(out), frame); print("->", out.name)
print("PASS facexlib RetinaFace", gpu_mem())

import cv2
import numpy as np
import os
import glob

with open(r".\nailong_vision\model\nailong.onnx", "rb") as f:
    net = cv2.dnn.readNetFromONNX(f.read())
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

outdir = r".\nailong_vision\test_results"
os.makedirs(outdir, exist_ok=True)

rows, cols, input_size, conf_thr = 2, 2, 224, 0.8

for img_path in sorted(glob.glob(r".\nailong_vision\test_images\*.jpg")):
    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    cw, ch = w // cols, h // rows
    cells, rects = [], []
    for r in range(rows):
        for c in range(cols):
            rects.append((c * cw, r * ch))
            cells.append(img[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw])
    blob = cv2.dnn.blobFromImages(cells, 1.0 / 255.0, (224, 224), (), True, False, cv2.CV_32F)
    net.setInput(blob)
    out = net.forward()
    found = 0
    name = os.path.basename(img_path)
    for i, (x, y) in enumerate(rects):
        l0, l1 = out[i, 0], out[i, 1]
        m = max(l0, l1)
        e0, e1 = np.exp(l0 - m), np.exp(l1 - m)
        conf = e1 / (e0 + e1)
        is_nailong = (l1 > l0) and conf >= conf_thr
        tag = "NAILONG" if is_nailong else "not"
        print(f"{name} cell({x},{y}): l0={l0:.2f} l1={l1:.2f} conf={conf:.3f} {tag}")
        if is_nailong:
            found += 1
            cv2.rectangle(img, (x, y), (x + cw, y + ch), (0, 0, 255), 4)
            label = f"Nailong {conf*100:.0f}%"
            cv2.putText(img, label, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    print(f"-> {name}: {found} cells hit")
    out_name = name.replace(".jpg", "_result.jpg")
    cv2.imwrite(os.path.join(outdir, out_name), img)
    print("saved", out_name)


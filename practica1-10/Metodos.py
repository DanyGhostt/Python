import cv2

# Ruta absoluta de tus archivos
ruta_base = "C:/Users/juand/OneDrive/Desktop/PracticasTopicos/practica1-10/"
template = cv2.imread(ruta_base + "template_002.jpg")

# Validar template
if template is None:
    exit()

# Redimensionar template al 50%
scale_percent = 50
template = cv2.resize(template, (
    int(template.shape[1] * scale_percent / 100),
    int(template.shape[0] * scale_percent / 100)
))
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

methods = [
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED
]

# ---------- 🖼️ 1. IMAGEN ----------
# image = cv2.imread(ruta_base + "imagen_002.jpg")
# if image is None:
#     exit()
# image = cv2.resize(image, (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100)))
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# for method in methods:
#     result = cv2.matchTemplate(image_gray, template_gray, method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     top_left = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
#     bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
#     img_copy = image.copy()
#     cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 2)
#     cv2.imshow("Template Matching", img_copy)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------- 🎥 2. VIDEO GRABADO ----------
# cap = cv2.VideoCapture(ruta_base + "video.mp4")
# if not cap.isOpened():
#     exit()
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.resize(frame, (int(frame.shape[1] * scale_percent / 100), int(frame.shape[0] * scale_percent / 100)))
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
#     _, _, _, max_loc = cv2.minMaxLoc(res)
#     top_left = max_loc
#     bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
#     cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
#     cv2.imshow("Video Matching", frame)
#     if cv2.waitKey(10) & 0xFF == 27:  # ESC para salir
#         break
# cap.release()
# cv2.destroyAllWindows()

# ---------- 📷 3. WEBCAM EN VIVO ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (int(frame.shape[1] * scale_percent / 100), int(frame.shape[0] * scale_percent / 100)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("Webcam Matching", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break
cap.release()
cv2.destroyAllWindows()

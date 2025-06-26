import cv2
import numpy as np

# ---------- Función para detectar color ----------
def detectar_color(hsv, contorno):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, -1)
    mean_val = cv2.mean(hsv, mask=mask)[:3]
    h = mean_val[0]

    if 0 <= h <= 10 or 160 <= h <= 180:
        return "Rojo"
    elif 11 <= h <= 25:
        return "Naranja"
    elif 26 <= h <= 34:
        return "Amarillo"
    elif 35 <= h <= 85:
        return "Verde"
    elif 100 <= h <= 130:
        return "Azul"
    elif 140 <= h <= 160:
        return "Rosa"
    else:
        return "Desconocido"

# ---------- Función para detectar forma ----------
def detectar_forma(contorno):
    approx = cv2.approxPolyDP(contorno, 0.04 * cv2.arcLength(contorno, True), True)
    lados = len(approx)

    if lados == 3:
        return "Triángulo"
    elif lados == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = float(w) / h
        return "Cuadrado" if 0.95 <= ratio <= 1.05 else "Rectángulo"
    elif lados == 5:
        return "Pentágono"
    elif lados == 6:
        return "Hexágono"
    elif lados > 6:
        return "Círculo"
    else:
        return "Desconocida"

# ----------- STREAMING DESDE IP WEBCAM --------------
url = 'http://192.168.100.4/video'  # ← CAMBIA esto a la IP de tu celular
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("No se pudo acceder al streaming.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500:
            forma = detectar_forma(contorno)
            color = detectar_color(hsv, contorno)

            x, y, w, h = cv2.boundingRect(contorno)
            texto = f"{forma} {color}"
            cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.drawContours(frame, [contorno], -1, (0, 255, 0), 2)

    cv2.imshow('Streaming con formas y colores', frame)
    cv2.imshow('Umbral', th)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()

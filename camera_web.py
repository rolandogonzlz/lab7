

import cv2
from ultralytics import YOLO


custom_model = YOLO(
    r"C:\Users\rolan\Downloads\Yachay\7_Septimo\IAYACHAY\Lab_7\runs\detect\custom_yolo_model\weights\best.pt"
)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(" No se pudo abrir la cámara.")
    exit()

print(" Cámara abierta. Presiona 'q' para salir.")


# Bucle principal de detección

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" No se pudo leer frame de la cámara.")
            break

        # Detección
        results = custom_model(frame, conf=0.4, imgsz=640)

        # Dibujar resultados directamente
        annotated = results[0].plot()

        # Mostrar en ventana
        cv2.imshow("Detección de reloj - YOLOv11", annotated)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(" Detenido por el usuario.")
            break

except Exception as e:
    print(f" Error durante la ejecución: {e}")

finally:
    
    # Liberar recursos
    
    cap.release()
    cv2.destroyAllWindows()
    print(" Cámara liberada y ventanas cerradas correctamente.")

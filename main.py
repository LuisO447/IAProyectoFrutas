#Importar librerias necesarias
from tkinter import *
from ultralytics import YOLO
import cv2

#Empezar a leer modelos

#model = YOLO("best.pt")
#model1 = YOLO("best2.pt")
model2 = YOLO("best.pt")

#Encender la cámara
cap = cv2.VideoCapture(0)

#Cuerpo principal del proyecto, bucle
while True:

    #Leer fotogramas con la cámara
    ret, frame = cap.read()

    #Leer modelos
    #resultado1 = model.predict(frame, imgsz = 640, conf = 0.50)
    #resultado2 = model1.predict(frame, imgsz=640, conf=0.45)
    resultado3 = model2.predict(frame, imgsz=640, conf=0.55)

    #Mostrar los resultados
    anotaciones = resultado3[0].plot()
    #anotaciones = resultado2[0].plot()

    #Mostrar los fotogramas
    cv2.imshow("Deteccion de Frutas", anotaciones)
    #ventanaTexto_ = Tk()
    #ventanaTexto_.title = ("Informacion")
    #etiqueta1 = Label(text=resultado3.plot(),font=("Arial",12))
    #etiqueta1.grid(column=0, row=1)

    #Detener el bucle de lectura y muestra
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
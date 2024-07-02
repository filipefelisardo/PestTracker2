import gradio as gr
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import io

def render_result(image, boxes):
    """Render the results with bounding boxes on the image."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])

        # Cor e o texto com base na classe
        if cls == 0:  # classe 0 = mosca
            color = (0, 255, 0)  # Verde para moscas
            label = f'Mosca: {conf:.2f}'
        else:
            color = (0, 0, 255)  # Vermelho para outros objetos
            label = f'Outro: {conf:.2f}'

        # desenhar o retangulo 
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Nivel de confiança
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def yoloV8_func(image):
    """This function performs YOLOv8 object detection on the given image."""
    try:
        # Parâmetros 
        image_size = 640
        conf_threshold = 0.25
        iou_threshold = 0.45

        # Upload do modelo YOLOv8 
        model_path = "best.pt" 
        model = YOLO(model_path)

        # Realiza a detecção de objetos na imagem de entrada usando o modelo YOLOv8
        results = model.predict(image,
                                conf=conf_threshold,
                                iou=iou_threshold,
                                imgsz=image_size)

        # # Mostra as informações dos objetos detectados (classe, coordenadas e probabilidade))
        box = results[0].boxes
        print("Object type:", box.cls)
        print("Coordinates:", box.xyxy)
        print("Probability:", box.conf)

        # Converte a imagem para numpy array se for necessário
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Mete na imagem caixas delimitadoras ao redor das moscas detectadas
        rendered_image = render_result(image, box)
        
        # Conta o numero de moscas
        num_objects = len(box)
        num_flies = sum(1 for b in box if int(b.cls[0]) == 0)
        num_others = num_objects - num_flies

        # Converta a imagem para o formato PIL para exibição no Gradio
        rendered_image_pil = Image.fromarray(cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB))
        
        detection_info = (f'Number of objects detected: {num_objects}\n'
                          f'Number of other objects detected: {num_others}')
    
        return rendered_image_pil, detection_info
    except Exception as e:
        error_message = f"Error processing the image: {str(e)}"
        print(error_message)
        return None, error_message

# Defina a interface do Gradio
inputs = gr.Image(type="filepath", label="Input Image")

outputs = [gr.Image(type="pil", label="Output Image"), gr.Textbox(label="Detection Info")]

title = "YOLOv8 101: Deteção de mosca"

yolo_app = gr.Interface(
    fn=yoloV8_func,
    inputs=inputs,
    outputs=outputs,
    title=title,
    cache_examples=True,
)

# Inicie a interface Gradio
yolo_app.launch(debug=True)

from flask import Flask, render_template, request
import os
import base64
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

class_mapping = {
    0: {'label': 'ship'}
}

def detect_and_visualize(img, model_path, class_mapping, confidence_threshold=0.25):
    model = YOLO(model_path)

    results = model.predict(source=img, conf=confidence_threshold)
    float_detections = results[0].boxes.xyxy.tolist()
    detections = [[int(value) for value in detection] for detection in float_detections]
    confidences = results[0].boxes.conf.tolist()
    float_classes = results[0].boxes.cls.tolist()
    classes = [int(value) for value in float_classes]

    ship_count = 0
    resized_img = cv2.resize(img, (800, 400))

    scaling_factor_x = 800 / img.shape[1]
    scaling_factor_y = 400 / img.shape[0]

    for i in range(len(detections)):
        box = detections[i]
        resized_box = [
            int(box[0] * scaling_factor_x),
            int(box[1] * scaling_factor_y),
            int(box[2] * scaling_factor_x),
            int(box[3] * scaling_factor_y)
        ]
        class_index = classes[i]
        class_info = class_mapping.get(class_index, {'label': 'unknown'})
        conf = confidences[i]
        if conf > 0.4:
            if class_info['label'] == 'ship':
                ship_count += 1

            class_label = class_info['label']
            cv2.putText(resized_img, f'{class_label} {conf:.3f}', (resized_box[0], resized_box[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.rectangle(resized_img, (resized_box[0], resized_box[1]), (resized_box[2], resized_box[3]), (255, 0, 255), 2)
    
    _, result_image = cv2.imencode('.jpg', resized_img)
    result_bytes = result_image.tobytes()

    return result_bytes, ship_count

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No selected file")

    if file and allowed_file(file.filename):
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        result_bytes, ship_count = detect_and_visualize(img, r"./best1.pt", class_mapping)
        
        return render_template('index.html', filename=f'data:image/jpg;base64,{base64.b64encode(result_bytes).decode()}', ship_count=ship_count, name=file.filename)

if __name__ == '__main__':
    app.run(debug=True)

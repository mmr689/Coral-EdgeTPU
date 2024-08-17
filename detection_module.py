import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

def load_labels(labels_path):
    with open(labels_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

def load_model(model_path):
    model = Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1', {'device': 'usb'})])
    model.allocate_tensors()
    return model

def preprocess_image(img_path, model_height, model_width):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (model_width, model_height))
    img_norm = img_resized.astype(np.uint8)
    img_batch = np.expand_dims(img_norm, axis=0)
    return img_batch, img.shape

def run_inference(model, image_batch):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], image_batch)
    model.invoke()
    boxes = model.get_tensor(output_details[0]['index'])  # Cajas delimitadoras
    classes = model.get_tensor(output_details[1]['index'])  # Clases detectadas
    scores = model.get_tensor(output_details[2]['index'])  # Puntuaciones de confianza
    num_detections = model.get_tensor(output_details[3]['index'])[0]  # Cantidad de detecciones
    return [boxes, classes, scores, int(num_detections)]

def get_predictions(results, image_shape, labels, score_threshold=0.5):
    boxes, classes, scores, num_detections = results
    width_img, height_img = image_shape[1], image_shape[0]
    predictions, plabels, pscores = [], [], []
    for i in range(int(num_detections)):
        score = scores[0][i]
        if score >= score_threshold:
            y_min, x_min, y_max, x_max = boxes[0][i]
            class_id = int(classes[0][i])
            plabels.append(labels[class_id])
            predictions.append((x_min * width_img, y_min * height_img, x_max * width_img, y_max * height_img))
            pscores.append(score)
    return predictions, plabels, pscores

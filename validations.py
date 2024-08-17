import cv2

from detection_module import load_labels, load_model, preprocess_image, run_inference, get_predictions
from iou_evaluator import evaluate_predictions
from xml_processor import load_and_extract_bboxes

# Rutas a los archivos necesarios
ind = 5
xml_path = f'squamata-dataset/annotations/xmls/squamata_{ind}.xml' # Ruta al xml de la imagen de entrada
img_path = f'squamata-dataset/images/squamata_{ind}.jpg' # Ruta a la imagen de entrada
model_path = 'models/squamata_edgetpu.tflite'
labels_path = 'models/squamata_labels.txt'

# INFERENCE
labels = load_labels(labels_path)
model = load_model(model_path)
img_batch, img_shape = preprocess_image(img_path, model.get_input_details()[0]['shape'][1], model.get_input_details()[0]['shape'][2])
results = run_inference(model, img_batch)
predictions, _, _ = get_predictions(results, img_shape, labels)

# XML GROUND TRUTH DATA
ground_truth = load_and_extract_bboxes(xml_path)


print(predictions)
print(ground_truth)
print('---')


# Evaluar predicciones
valid_predictions, false_positives, false_negatives = evaluate_predictions(predictions, ground_truth)

# Imprimir resultados
print("Valid Predictions:", valid_predictions)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)


img = cv2.imread(img_path)
for (xmin, ymin, xmax, ymax) in predictions:
    start_point = (int(xmin), int(ymin))
    end_point = (int(xmax), int(ymax))
    cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
for (xmin, ymin, xmax, ymax) in ground_truth:
    start_point = (int(xmin), int(ymin))
    end_point = (int(xmax), int(ymax))
    cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)

cv2.imwrite('img_predict.jpg', img)
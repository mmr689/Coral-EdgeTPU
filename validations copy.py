"""
Proceso de evaluación de una foto"""
import cv2

from detection_module import load_labels, load_model, preprocess_image, run_inference, get_predictions
from iou_evaluator import evaluate_predictions
from xml_processor import load_and_extract_bboxes

iou_threshold = 0.5

test_paths = 'squamata-dataset/annotations/test.txt'
model_path = 'models/squamata_edgetpu.tflite'
labels_path = 'models/squamata_labels.txt'

labels = load_labels(labels_path)
model = load_model(model_path)

# Abre el archivo en modo lectura y extrae nombre archivo
with open(test_paths, 'r') as file:
    lines = file.readlines()
names = [line.split()[0] for line in lines]

tp_count, fp_count, fn_count = 0, 0, 0
for name in names:
    # Rutas a los archivos necesarios
    xml_path = f'squamata-dataset/annotations/xmls/{name}.xml' # Ruta al xml de la imagen de entrada
    img_path = f'squamata-dataset/images/{name}.jpg' # Ruta a la imagen de entrada

    # INFERENCE
    img_batch, img_shape = preprocess_image(img_path, model.get_input_details()[0]['shape'][1], model.get_input_details()[0]['shape'][2])
    results = run_inference(model, img_batch)
    predictions, _, _ = get_predictions(results, img_shape, labels)

    # XML GROUND TRUTH DATA
    ground_truth = load_and_extract_bboxes(xml_path)

    # Evaluar predicciones
    tp, fp, fn = evaluate_predictions(predictions, ground_truth, iou_threshold=iou_threshold)

    tp_count += len(tp)
    fp_count += len(fp)
    fn_count += len(fn)

print('Total fotos', len(names))
print('TP:', tp_count)
print('FP:', fp_count)
print('FN:', fn_count)

precision = tp_count / (tp_count + fp_count)
recall = tp_count / (tp_count + fn_count)
print(f'Precisión: {precision:.2f}')
print(f'Recall: {recall:.2f}')

precision_recall = [(precision, recall)]
precision_at_recalls = [pr[0] for pr in precision_recall]
print(precision_at_recalls)

map_score = sum(precision_at_recalls) / len(precision_at_recalls)
print(f'mAP: {map_score:.2f}')
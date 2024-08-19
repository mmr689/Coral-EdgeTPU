"""
Inferencia con métricas. Basado en inference.py
Cargamos una imagen, comparamos con ground truth, evaluamos y dibujamos bounding boxes.
    - img_predict_metrics_gtruth:
        · Verde predicciones,
        · Rojo ground truth.
    - img_predict_metrics_confmatrix:
    · Verde TP,
    · Negro: TP redundantes (en métrica final esto es un FP),
    · Rojo: FP,
    · Amarillo: FN.
"""

import cv2

from utils.detection_module import load_labels, load_model, preprocess_image, run_inference, get_predictions
from utils.iou_evaluator import evaluate_predictions
from utils.xml_processor import load_and_extract_bboxes

def main():
    # Rutas a los archivos necesarios
    ind = 299
    xml_path = f'squamata-dataset/annotations/xmls/squamata_{ind}.xml' # Ruta al xml de la imagen de entrada
    img_path = f'squamata-dataset/images/squamata_{ind}.jpg' # Ruta a la imagen de entrada
    model_path = 'models/squamata_edgetpu.tflite'
    labels_path = 'models/squamata_labels.txt'

    # Carga de modelo
    labels = load_labels(labels_path)
    model = load_model(model_path, edgetpu=True)

    # Inference
    img_batch, img_shape = preprocess_image(img_path, model.get_input_details()[0]['shape'][1], model.get_input_details()[0]['shape'][2])
    results = run_inference(model, img_batch)
    predictions, _, _ = get_predictions(results, img_shape, labels)

    # XML GROUND TRUTH DATA
    ground_truth = load_and_extract_bboxes(xml_path)

    # Evaluar predicciones
    tp, red_tp, fp, fn = evaluate_predictions(predictions, ground_truth, iou_threshold=0.5)

    # Imprimir resultados
    print("Valid Predictions:", tp)
    print("Redundant Predictions:", red_tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)

    img = cv2.imread(img_path)
    for (xmin, ymin, xmax, ymax) in predictions:
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
    for (xmin, ymin, xmax, ymax) in ground_truth:
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)
    
    cv2.imwrite('results/img_predict_metrics_gtruth.jpg', img)

    img = cv2.imread(img_path)
    for (xmin, ymin, xmax, ymax), iou in tp:
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
    for (xmin, ymin, xmax, ymax), iou in red_tp:
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        cv2.rectangle(img, start_point, end_point, (0, 0, 0), 2)
    for (xmin, ymin, xmax, ymax), iou in fp:
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)
    for (xmin, ymin, xmax, ymax), iou in fn:
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        cv2.rectangle(img, start_point, end_point, (255, 0, 255), 2)

    cv2.imwrite('results/img_predict_metrics_confmatrix.jpg', img)
    

if __name__ == "__main__":
    
    main()

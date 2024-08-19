"""
Inferencia de múltiples imágenes (test) con métricas. Basado en inference.py
Cargamos una imagen, comparamos con ground truth y evaluamos.
"""

from utils.detection_module import load_labels, load_model, preprocess_image, run_inference, get_predictions
from utils.iou_evaluator import evaluate_predictions
from utils.xml_processor import load_and_extract_bboxes

def main():
    iou_threshold = 0.5
    # Rutas a los archivos necesarios estáticas
    model_path = 'models/squamata_edgetpu.tflite'
    labels_path = 'models/squamata_labels.txt'
    test_paths = 'squamata-dataset/annotations/test short.txt'

    # Obtenemos los archivos de text en test.txt
    with open(test_paths, 'r') as file:
        lines = file.readlines()
    names = [line.split()[0] for line in lines]

    # Carga de modelo
    labels = load_labels(labels_path)
    model = load_model(model_path, edgetpu=True)

    tp_count, red_tp_count, fp_count, fn_count = 0, 0, 0, 0
    for name in names:
        # Rutas a los archivos necesarios
        xml_path = f'squamata-dataset/annotations/xmls/{name}.xml' # Ruta al xml de la imagen de entrada
        img_path = f'squamata-dataset/images/{name}.jpg' # Ruta a la imagen de entrada
    
        # Inference
        img_batch, img_shape = preprocess_image(img_path, model.get_input_details()[0]['shape'][1], model.get_input_details()[0]['shape'][2])
        results = run_inference(model, img_batch)
        predictions, _, _ = get_predictions(results, img_shape, labels)

        # XML GROUND TRUTH DATA
        ground_truth = load_and_extract_bboxes(xml_path)

        # Evaluar predicciones
        tp, red_tp, fp, fn = evaluate_predictions(predictions, ground_truth, iou_threshold=iou_threshold)

        tp_count += len(tp)
        red_tp_count += len(red_tp)
        fp_count += len(fp)
        fn_count += len(fn)

    print('Total fotos', len(names))
    print('TP:', tp_count)
    print('Red TP', red_tp_count)
    print('FP:', fp_count)
    print('FN:', fn_count)

    precision = tp_count / (tp_count + fp_count + red_tp_count)
    recall = tp_count / (tp_count + fn_count)
    print(f'Precisión: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    precision_recall = [(precision, recall)]
    precision_at_recalls = [pr[0] for pr in precision_recall]
    map_score = sum(precision_at_recalls) / len(precision_at_recalls)
    print(f'mAP: {map_score:.4f}')
    

if __name__ == "__main__":
    
    main()

"""
Inferencia básica. Cargamos una imagen y dibujamos bounding boxes
"""

import cv2

from utils.detection_module import load_labels, load_model, preprocess_image, run_inference, get_predictions

def main():

    # Rutas a los archivos necesarios
    ind = 299
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

    img = cv2.imread(img_path)
    for (xmin, ymin, xmax, ymax) in predictions:
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
    
    cv2.imwrite('results/img_predict.jpg', img)
    

if __name__ == "__main__":
    
    main()

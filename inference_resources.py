"""
Inferencia de múltiples imágenes (test) para conocer los recursos empleados. Basado en inference.py
Monitoreamos todo el proceso y ponemos flags en los puntos que queremos conocer para finalmente generar un csv
con los datos. de aquí podemos conocer tiempos ejecución, cpu y memoria
"""
import threading
import psutil
import time
import datetime
import pandas as pd

from utils.detection_module import load_labels, load_model, preprocess_image, run_inference, get_predictions

class MonitorState:
    def __init__(self, sampling_interval=1):
        self.current_marker = None
        self.records = []
        self.monitor_active = True
        self.sampling_interval = sampling_interval

def resource_monitor(state, interval=1):
    process = psutil.Process()
    while state.monitor_active:
        cpu = process.cpu_percent(interval=None) 
        memory = process.memory_info().rss / (1024 * 1024)  # Convertir bytes a MB
        timestamp = datetime.datetime.now()
        state.records.append((timestamp, state.current_marker, cpu, memory))
        time.sleep(state.sampling_interval)

def main_processing(state):

    state.current_marker = "-"

    test_paths = 'squamata-dataset/annotations/test short.txt'
    model_path = 'models/squamata.tflite'
    labels_path = 'models/squamata_labels.txt'

    # Obtenemos los archivos de text en test.txt
    with open(test_paths, 'r') as file:
        lines = file.readlines()
    names = [line.split()[0] for line in lines]

    # Carga de modelo
    labels = load_labels(labels_path)
    state.current_marker = "load_model"
    model = load_model(model_path, edgetpu=False)
    state.current_marker = "-"
    
    for name in names:
        # Rutas a los archivos necesarios
        img_path = f'squamata-dataset/images/{name}.jpg' # Ruta a la imagen de entrada

        # INFERENCE
        img_batch, img_shape = preprocess_image(img_path, model.get_input_details()[0]['shape'][1], model.get_input_details()[0]['shape'][2])
        state.current_marker = "run_inference"
        results = run_inference(model, img_batch)
        state.current_marker = "get_predictions"
        _, _, _ = get_predictions(results, img_shape, labels)
        state.current_marker = "-"
    
    state.monitor_active = False

if __name__ == "__main__":
    state = MonitorState(sampling_interval=0.1)
    monitor_thread = threading.Thread(target=resource_monitor, args=(state,))
    monitor_thread.start()
    
    main_processing(state)
    monitor_thread.join()  # Esperar a que el hilo de monitoreo finalice

    # Convertir los registros a DataFrame y guardar en CSV
    df = pd.DataFrame(state.records, columns=['Timestamp', 'Marker', 'CPU_Usage', 'Memory_Usage'])
    df.to_csv("results/data/results_resources.csv", index=False)

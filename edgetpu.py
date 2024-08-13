from tflite_runtime.interpreter import Interpreter, load_delegate
import cv2
import numpy as np

# Rutas a los archivos necesarios
img_path = '/home/pi/Coral_EdgeTPU-RPi/assets/ffn-test.jpg'  # Ruta a la imagen de entrada
model_path = '/home/pi/Coral_EdgeTPU-RPi/models/ffn-test-ssd-mbl-v1.tflite'  # Ruta al modelo de TensorFlow Lite
labels_path = '/home/pi/Coral_EdgeTPU-RPi/models/ffn_labels.txt'  # Ruta al archivo de etiquetas

# Cargar las etiquetas desde el archivo
with open(labels_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Cargar el modelo TFLite y configurar la TPU como delegado
model = Interpreter(model_path,
                    experimental_delegates=[load_delegate('libedgetpu.so.1', options={'device': 'usb'})])
model.allocate_tensors()  # Reservar memoria para los tensores

# Obtener detalles sobre los tensores de entrada y salida
input_details = model.get_input_details()
output_details = model.get_output_details()
_, model_height, model_width, _ = input_details[0]['shape']
max_detections = output_details[0]['shape'][2]  # Máximo de detecciones que el modelo puede realizar
print(model_height, model_width, max_detections)  # Mostrar dimensiones esperadas de entrada y máx. detecciones

# Cargar y preprocesar la imagen
img = cv2.imread(img_path)  # Leer la imagen
height_img, width_img, _ = img.shape  # Obtener dimensiones de la imagen
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
img_resized = cv2.resize(img_rgb, (model_width, model_height))  # Cambiar tamaño para adaptarse al modelo
img_norm = img_resized.astype(np.uint8)  # Normalizar los valores de píxel
img_batch = np.expand_dims(img_norm, axis=0)  # Añadir una dimensión para el batch

# Ejecutar el modelo con la imagen
model.set_tensor(input_details[0]['index'], img_batch)
model.invoke()
results = model.get_tensor(output_details[0]['index'])
print(results.shape)

# Imprimir los nombres de los tensores de salida para entender lo que devuelven
for output in output_details:
    print(output['name'])

# Recuperar las detecciones del modelo
boxes = model.get_tensor(output_details[0]['index'])  # Cajas delimitadoras
classes = model.get_tensor(output_details[1]['index'])  # Clases detectadas
scores = model.get_tensor(output_details[2]['index'])  # Puntuaciones de confianza
num_detections = model.get_tensor(output_details[3]['index'])[0]  # Cantidad de detecciones
num_detections = int(num_detections)  # Convertir a entero

print("Número de detecciones:", num_detections)
for i in range(num_detections):
    score = scores[0][i]  # Puntuación de confianza de la detección actual
    class_id = int(classes[0][i])  # ID de la clase detectada
    label = labels[class_id]  # Etiqueta de la clase

    if score >= 0.5:  # Umbral de confianza para mostrar detecciones
        y_min, x_min, y_max, x_max = boxes[0][i]

        print(f"Detección {i+1}:")
        print(f"  Clase (ID): {label} ({class_id})")
        print(f"  Confianza: {score:.2f}")
        print(f"  BBox: [y_min: {y_min}, x_min: {x_min}, y_max: {y_max}, x_max: {x_max}]")
        print("----------")
        
        # Dibujar un rectángulo alrededor de cada detección
        start_point = (int(x_min * width_img), int(y_min * height_img))
        end_point = (int(x_max * width_img), int(y_max * height_img))
        color = (255, 0, 0)  # Color del rectángulo, RGB
        thickness = 2  # Grosor del rectángulo
        cv2.rectangle(img, start_point, end_point, color, thickness)

# Guardar la imagen con las detecciones
cv2.imwrite('img_predict.jpg', img)
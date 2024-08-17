# Entrenar con tu propio conjunto de datos

Este README explica cómo entrenar modelos de detección de objetos personalizados para ejecutarse en la EdgeTPU de Google Coral.

## Estructura de los datos de trabajo

A continuación se define cómo debe ser la estructura de datos del proyecto. Es importante cumplirla ya que estamos adaptando un código para que funcione con nuestros propios datos, por lo tanto debemos seguir las reglas del que lo creo.

Una de las cosas que debemos entender es que tendremos un archivo (imagen, trimap, xml) por cada objeto que aparezca en una imagen. Es decir, una imagen con diferentes objetos o el mismo repetid, se convertirá en diferentes archivos de trabajo, asumiendo que la imagen estará repetida.

### Estructura de los datos

Es esencial preparar correctamente el conjunto de datos para el entrenamiento del modelo. La estructura de directorios debe ser la siguiente dentro de la carpeta del proyecto:

```
/proyecto
│
├── images/             # Contiene todas las imágenes para entrenar y evaluar el modelo.
└── annotations/      
    ├── xmls/           # Archivos XML con las anotaciones de las imágenes.
    ├── trimaps/        # Imágenes trimap que se utilizan para segmentación específica, no los utilizamos pero debemos crearlos.
    ├── list.txt        # Lista completa de imágenes.
    ├── trainval.txt    # Lista de imágenes utilizadas para el entrenamiento y validación.
    └── test.txt        # Lista de imágenes utilizadas para pruebas.
└── pet_label_map.pbtxt # Archivo que define las clases de trabajo.
```

### Descripción de los componentes

- **images/**: Este directorio debe contener todas las imágenes que se utilizarán en el entrenamiento y la evaluación del modelo. Cada imagen debe estar en un formato compatible, como JPG o PNG y deberá llamarse con el siguiente formato, `label_1.ext`. Si hay diferentes objetos en la imagen, crearemos varias veces la imagen con diferentes nombres, ya sea en el identificador o en la etiqueta.

- **annotations/**:
  - **xmls/**: Contiene archivos XML para cada imagen, los cuales especifican las coordenadas de los objetos dentro de cada imagen, esencial para el entrenamiento de detección de objetos.
    ```bash
    <annotation>
        <folder>folder_name</folder>            # Valor informativo
        <filename>label_1.jpg</filename>        # Nombre de la imagen asociada
        <source>
            <database>Unknown</database>        # Valor informativo
            <annotation>Unknown</annotation>    # Valor informativo
            <image>Unknown</image>              # Valor informativo
        </source>
            <size>
                <width>4656</width>             # Ancho de la imagen
                <height>3496</height>           # Alto de la imagen
                <depth>3</depth>                # Profundidad/Canales de trabajo
            </size>
        <segmented>0</segmented>                # Valor a obviar y mantener
        <object>
            <name>squamata</name>               # Etiqueta
            <pose>Frontal</pose>                # Valor a obviar y mantener
            <truncated>0</truncated>            # Valor a obviar y mantener
            <occluded>0</occluded>              # Valor a obviar y mantener
            <bndbox>                            # Coordenadas de la caja, puede ser float.
                <xmin>2757.85</xmin>
                <ymin>2348.92</ymin>
                <xmax>3088.94</xmax>
                <ymax>2482.97</ymax>
            </bndbox>
            <difficult>0</difficult>            # Valor a obviar y mantener
        </object>
    </annotation>
    ```
  - **trimaps/**: Contiene imágenes "trimap" que se utilizan en tareas de segmentación. Como nosotros hacemos detección no sirven para nada, por lo tanto basta añadir una imagen negra con su nombre asociado. Importante, parece que el formato tiene que ser con extensión `.png`. Es decir, `label_1.png`.
  - **list.txt**: Archivo de texto que enumera todas las imágenes en el conjunto de datos y define la clase y otros dos identificadores que no damos uso. Quedaría algo similar a:
    ```bash
    dog_1 1 1 1
    dog_2 1 1 1
    cat_1 2 1 1
    fox_1 3 1 1
    cat_2 2 1 1
    dog_3 1 1 1
    ```
  - **trainval.txt**: Lista de imágenes seleccionadas para entrenamiento y validación. Estos archivos son usados para dividir el dataset de manera efectiva durante el entrenamiento. El formato es el mismo que en **list.txt**.
  - **test.txt**: Lista de imágenes usadas para pruebas finales, permitiendo evaluar la precisión del modelo en datos no vistos durante el entrenamiento. El formato es el mismo que en **list.txt**.
  - **pet_label_map.pbtxt**: Archivo deonde definimos las clases a detectar. Su formato será:
  ```bash
    item{
        id:1
        name: 'label1'
    }
    item{
        id:2
        name: 'label2'
    }
  ```

## Creación del modelo

### Clonar y preparar el entorno

```bash
# Clona el repositorio y construye la imagen de Docker
git clone https://github.com/google-coral/tutorials.git
cd tutorials/docker/object_detection
sudo docker build . -t detect-tutorial-tf1

# Prepara el directorio de salida
DETECT_DIR=${PWD}/out && mkdir -p $DETECT_DIR
sudo docker run --name edgetpu-detect --rm -it --privileged -p 6006:6006 \
--mount type=bind,src=${DETECT_DIR},dst=/tensorflow/models/research/learn_pet \
detect-tutorial-tf1
```

### Preparar datos en Docker

```bash
# En el Docker, prepara la estructura del directorio para el dataset
./prepare_checkpoint_and_dataset.sh --network_type mobilenet_v1_ssd --train_whole_model false
mkdir -p /tensorflow/models/research/learn_pet/pet/{images,annotations}
```

### Transferir conjunto de datos al contenedor Docker

```bash
# Copia tus datos de imágenes y anotaciones al Docker
sudo docker cp /home/user/PATH/images DOCKER_SESSION_ID:/tensorflow/models/research/learn_pet/pet/
sudo docker cp /home/user/PATH/annotations DOCKER_SESSION_ID:/tensorflow/models/research/learn_pet/pet/
sudo docker cp /home/user/PATH/pet_label_map.pbtxt DOCKER_SESSION_ID:/tensorflow/models/research/learn_pet/pet/
sudo docker cp /home/user/PATH/pet_label_map.pbtxt DOCKER_SESSION_ID:/tensorflow/models/research/object_detection/data/
```

### Entrenamiento y conversión del modelo

```bash
# Entrena y convierte el modelo para EdgeTPU
python object_detection/dataset_tools/create_pet_tf_record.py \
--label_map_path=/tensorflow/models/research/learn_pet/pet/pet_label_map.pbtxt \
--data_dir=/tensorflow/models/research/learn_pet/pet/ --output_dir=/tensorflow/models/research/learn_pet/pet/

NUM_TRAINING_STEPS=500 && NUM_EVAL_STEPS=100

./retrain_detection_model.sh --num_training_steps ${NUM_TRAINING_STEPS} --num_eval_steps ${NUM_EVAL_STEPS}
./convert_checkpoint_to_edgetpu_tflite.sh --checkpoint_num 500

# Instala el compilador de EdgeTPU
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
apt update && apt-get install edgetpu-compiler

# Compila y renombra el modelo compilado
edgetpu_compiler output_tflite_graph.tflite
mv output_tflite_graph_edgetpu.tflite name.tflite
```

Este proceso debería resultar en un modelo entrenado personalizado listo para ser ejecutado en una EdgeTPU. Concretamente `tutorials/docker/object_detection/out/models` podremos encontrar el modelo, cuantificado y preparado para trabajar en la EdgeTPU, bajo el nombre `name.tflite`. El archivo `label.txt` que encontraremos no sirve, ya que será el que realmente se habría creado si no hubieramos alterado el proceso, deberemos crearlo nosotros.
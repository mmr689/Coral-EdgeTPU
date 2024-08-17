# Pasos seguidos para configurar la Raspberri Pi 4

En mi caso he empezado montando una imagen de cero. Empezamos comprobando la versión del SO, `cat /etc/os-release`,

```bash
PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
NAME="Debian GNU/Linux"
VERSION_ID="12"
VERSION="12 (bookworm)"
VERSION_CODENAME=bookworm
ID=debian
HOME_URL="https://www.debian.org/"
SUPPORT_URL="https://www.debian.org/support"
BUG_REPORT_URL="https://bugs.debian.org/"
```

también comprobamos la versión del Kernel, `uname -a`,

```bash
Linux raspberrypi 6.6.31+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.6.31-1+rpt1 (2024-05-29) aarch64 GNU/Linux
```

Comprobamos la versión de Python instalada, `python --version`.

```bash
Python 3.11.2
```

## Instalar dependencias

El primer paso a sido actualizar la SBC.

```bash
sudo apt update
sudo apt full-upgrade
```

Una vez está a punto, he creado una carpeta de trabajo, un entorno virtual y he instalado la librerías necesarias para poder ejecutar el modelo de los tutoriales de Coral. Cómo los modelos se crearon con Tensorflow 1 y Numpy 1, debemos instalar una versión de Numpy inferior a la 2 (a pesar de estar desfasado).

Además, para poder trabajar con imágenes he añadido la librería de OpenCV.

```bash
mkdir Coral_EdgeTPU-RPi
cd Coral_EdgeTPU-RPi/

python3 -m venv .venv
source .venv/bin/activate

pip install tflite-runtime==2.14.0
pip install numpy==1.26.4
pip install opencv-python==4.10.0.82
sudo apt install libgl1-mesa-glx # Necesario para emplear Opencv
```

Hasta aquí ya podemos hacer la inferencia de un modelo `.tflite` general, es decir, sin USB acelerador.

Podemos comprobarlo con:

```python
from tflite_runtime.interpreter import Interpreter

# Load the model
model = Interpreter('/path_to_model/model.tflite')
model.allocate_tensors()

# Get model details
input_details = model.get_input_details()
output_details = model.get_output_details()
_, height, width, _ = input_details[0]['shape']
print(height, width)
```

## Instalar USB EdgeTPU

Para emplear el USB Acelerador en tu Raspberry Pi que utiliza una arquitectura ARM de 64 bits (aarch64), deberás adaptar un poco los pasos oficiales que proporciona [Coral](https://coral.ai/docs/accelerator/get-started/#runtime-on-linux), combinándolos con la información del [post no oficial](https://github.com/feranick/libedgetpu/releases) para obtener el paquete correcto para tu sistema. Aquí te muestro cómo quedarían los pasos integrando la fuente no oficial para el driver adecuado:

1. **Añadir el repositorio de paquetes de Debian de Coral alsistema:**
   ```
   echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
   ```

2. **Añadir la llave GPG del repositorio de Coral:**
   ```
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   ```

3. **Actualizar la lista de paquetes:**
   ```
   sudo apt update
   ```

4. **Descargar e instalar el paquete específico para ARM64 desde el repositorio no oficial:**
   ```
   wget https://github.com/feranick/libedgetpu/releases/download/v16.0TF2.15.1-1/libedgetpu1-std_16.0tf2.15.1-1.bookworm_arm64.deb
   sudo apt install ./libedgetpu1-std_16.0tf2.15.1-1.bookworm_arm64.deb
   ```

   Lo guardamos en una carpeta llamada `pck` por si a futuro nos hiciese falta.

5. **Conectar el USB Accelerator a la Raspberry Pi. Si ya estaba conectado anteriormente, desconectarlo y colver a conectar para que la regla udev recién instalada pueda tener efecto.**

Comprobamos que efectivamente funciona,

```python
from tflite_runtime.interpreter import Interpreter, load_delegate

# Load the model
model = Interpreter('/PATH/model_edgetpu.tflite',
        experimental_delegates=[load_delegate('libedgetpu.so.1', options={'device': 'usb'})]) # Línea modificada par poder emplear el modelo adaptado a las TPUs de Coral
model.allocate_tensors()

# Get model details
input_details = model.get_input_details()
output_details = model.get_output_details()
_, height, width, _ = input_details[0]['shape']
print(height, width)
```

Para ver una ejecución completa de un modelo y el USB EdgeTPU, acudir al script, `edgetpu.py`. Allí podremos observar cómo se carga el modelo, cómo se preparan los datos y finalmente como se intepreta la salida de la inferencia.

Los modelos de ejemplo está disponibles en la [web de Coral](https://coral.ai/models/object-detection/).

Todo lo anterior se ha testeado con los siguientes modelos.

<table>
   <tr>
      <th></th>
      <th>ssd-mblnet-v1*</th>
      <th>ssd-mblnet-v2*</th>
      <th>ssd-mbldet</th>
   </tr>
   <tr>
      <th>Tensoflow Version</th>
      <th>1</th>
      <th>1</th>
      <th>1</th>
   </tr>
   <tr>
      <th>Train Dataset</th>
      <th>COCO</th>
      <th>COCO</th>
      <th>COCO</th>
   </tr>
   <tr>
      <th>Labels</th>
      <th>80</th>
      <th>80</th>
      <th>80</th>
   </tr>
   <tr>
      <th>Width x Height</th>
      <th>300x300</th>
      <th>300x300</th>
      <th>320x320</th>
   </tr>
   <tr>
      <th>Max detections</th>
      <th>4</th>
      <th>4</th>
      <th>4</th>
   </tr>
   <tr>
      <th>Formato datos</th>
      <th>UINT8</th>
      <th>UINT8</th>
      <th>UINT8</th>
   </tr>
   <tr>
      <th>Shape</th>
      <th>(1, 20, 4)</th>
      <th>(1, 20, 4)</th>
      <th>(1, 100, 4)</th>
   </tr>
   <tr>
      <th>NMS</th>
      <th>Sí</th>
      <th>Sí</th>
      <th>Sí</th>
   </tr>
</table>

- *ssd-mblnet-v1: ssd_mobilenet_v1_coco_quant_postprocess_edgetpu
- *ssd-mblnet-v2: ssd_mobilenet_v2_coco_quant_postprocess_edgetpu
- *ssd-mbldet: ssdlite_mobiledet_coco_qat_postprocess_edgetpu

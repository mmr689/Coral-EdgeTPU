# Pasos seguidos para configurar la Rock 4C+

Estos pasos se basan en los pasos seguidos para configurar la Raspberry Pi, por lo tanto serán más concisos.

## Características del hardware

Empezamos comprobando la versión del SO, `cat /etc/os-release`,

```bash
PRETTY_NAME="Debian GNU/Linux 11 (bullseye)"
NAME="Debian GNU/Linux"
VERSION_ID="11"
VERSION="11 (bullseye)"
VERSION_CODENAME=bullseye
ID=debian
HOME_URL="https://www.debian.org/"
SUPPORT_URL="https://www.debian.org/support"
BUG_REPORT_URL="https://bugs.debian.org/"
```

tambien comprobamos la versión del Kernel, `uname -a`,

```bash
Linux rock-4c-plus 5.10.110-20-rockchip #e0ac49d1b SMP Tue Sep 26 08:20:36 UTC 2023 aarch64 GNU/Linux
```

Conocemos la versión de python instalada `python --version`

```bash
Python 3.9.2
```

### Actualizamos SO e instalamos dependencias

```bash
sudo apt update
sudo apt full-upgrade

sudo apt install git
sudo apt install python3-venv

git clone https://github.com/mmr689/Coral_EdgeTPU-RPi.git

python3 -m venv .venv
source .venv/bin/activate

pip install tflite-runtime==2.13.0 # No está disponible la versión 2.14.0 como en la RPi, lo importante es que sea superior a 2.11
pip install numpy==1.26.4
pip install opencv-python==4.10.0.82
sudo apt install libgl1-mesa-glx # Necesario para emplear Opencv
```

Efectivamente hasta aquí ya funciona las inferencias con modelos `.tflite` sin emplear el USB de Coral.

### Instalar USB EdgeTPU

Los pasos son similares a la Raspberry Pi, seguimos teniendo un sistema `arm64` pero en este caso el SO de Debian es el Bullseye. Por lo tanto:

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
wget https://github.com/feranick/libedgetpu/releases/download/v16.0TF2.15.1-1/libedgetpu1-std_16.0tf2.15.1-1.bullseye_arm64.deb
sudo apt install ./libedgetpu1-std_16.0tf2.15.1-1.bullseye_arm64.deb
```

*Lo guardamos en una carpeta llamada `pck` por si a futuro nos hiciese falta.*

Conectamos el USB a la Rock y lo probamos. **IMPORTANTE: SOLO FUNCIONA EN LOS USB 2.0 (NEGROS), LOS USB 3.0 (AZULES) DAN ERROR.**
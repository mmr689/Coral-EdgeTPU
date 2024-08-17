![Banner](assets/banner.png)

**Disclaimer: Este README no está acabado y por ahora es una guía para entenderla yo.**

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

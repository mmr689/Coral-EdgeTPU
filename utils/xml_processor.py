import xml.etree.ElementTree as ET

def load_and_extract_bboxes(xml_path):
    """Carga un archivo XML y extrae las bounding boxes, opcionalmente dibuja en una imagen.
    
    Args:
        xml_path (str): Ruta al archivo XML.

    Returns:
        list of tuple: Lista de tuplas con las bounding boxes (xmin, ymin, xmax, ymax).
    """
    # Cargar el XML desde el archivo
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Encontrar todos los objetos en el XML y extraer las bounding boxes
    ground_truth = []
    for obj in root.findall('.//object'):
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text
        ground_truth.append((float(xmin), float(ymin), float(xmax), float(ymax)))

    return ground_truth

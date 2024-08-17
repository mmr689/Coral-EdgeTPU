from tflite_runtime.interpreter import Interpreter, load_delegate

# Load the model
model = Interpreter('models/squamata_edgetpu.tflite',
        experimental_delegates=[load_delegate('libedgetpu.so.1', options={'device': 'usb'})]) # Línea modificada par poder emplear el modelo adaptado a las TPUs de Coral
model.allocate_tensors()

# Get model details
input_details = model.get_input_details()
output_details = model.get_output_details()
_, height, width, _ = input_details[0]['shape']
print(height, width)
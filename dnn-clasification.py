import cv2
import numpy as np

#abrimos el archivo de texto que contiene todos los nombres de clases en modo lectura y los dividimos usando cada nueva línea.
with open('classification_classes_ILSVRC2012.txt', 'r') as f:
   image_net_names = f.read().split('\n')
   
#Sin embargo, solo necesitamos el nombre de cada línea. Eso es lo que hace la segunda línea de código. Para cada elemento en la lista image_net_names, dividimos los elementos usando una coma (,) como delimitador y solo mantenemos el primero de esos elementos 
class_names = [name.split(',')[0] for name in image_net_names]#output etiquetas
print("class name :) ",class_names)

# load the neural network model
model = cv2.dnn.readNet(model='DenseNet_121.caffemodel', config='DenseNet_121.prototxt', framework='Caffe')

# load the image from disk
image = cv2.imread('tiger.jpg')

#scalefactor: este valor escala la imagen según el valor proporcionado. Tiene un valor predeterminado de 1, lo que significa que no se realiza ningún escalado.
#(224 × 224),la mayoría de los modelos de clasificación entrenados en el conjunto de datos de ImageNet solo esperan este tamaño.
#Todos los modelos de aprendizaje profundo esperan entradas en lotes. Sin embargo, aquí sólo tenemos una imagen.
blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))
#print(blob.shape)#(1, 3, 224, 224)
#print(type(blob))#<class 'numpy.ndarray'>

model.setInput(blob)

outputs = model.forward()#(1, 1000, 1, 1)las salidas son un tensor con cuatro dimensiones, donde la segunda dimensión representa el numero de clases.

final_outputs = outputs[0]#se extrae la primera dimension (output: [[-4.43995774e-01]]) longitud:1000
#print("final outputs shape",final_outputs.shape)#(1000, 1, 1)
final_outputs = final_outputs.reshape(1000, 1)#indica que tiene 1000 filas para las 1000 etiquetas. Cada fila contiene la puntuación correspondiente a la etiqueta de clase, que se parece a la siguiente.
#[[-1.44623446e+00]

label_id = np.argmax(final_outputs)#extraemos el índice de etiqueta más alto
#print("indice de etiqueta mas alto",label_id)#292 (tiger)
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))#estamos convirtiendo las puntuaciones a probabilidades softmax usando np.exp(final_outputs) / np.sum(np.exp(final_outputs)). Luego multiplicamos el puntaje de probabilidad más alto por 100 para obtener el porcentaje de puntaje previsto.output ([[4.78046158e-10])
# get the final highest probability
final_prob = np.max(probs) * 100.#multiplicamos el puntaje de probabilidad más alto por 100 para obtener el porcentaje de puntaje previsto.
#print("final_prob ",final_prob)# output (91.02953672409058)

out_name = class_names[label_id]#output tiger
out_text = f"{out_name}, {final_prob:.3f}"#output tiger, 91.030
# put the class name text on top of the image
cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)
#cv2.imwrite('result_image.jpg', image)
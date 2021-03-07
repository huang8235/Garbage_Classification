from ast import increment_lineno
import matplotlib
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import activations
from tensorflow.keras.models import model_from_json

#模型加载
json_file=open('model_json.json')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)
model.load_weights('model_weight.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['acc'])

(train_image,train_label),(test_image,test_label)=tf.keras.datasets.fashion_mnist.load_data()
train_image=train_image/255.0  #归一化
test_image=test_image/255.0

model.evaluate(test_image,test_label)
predict=model.predict(test_image)
print(np.argmax(predict[0]))

plt.imshow(test_image[0])
plt.show()
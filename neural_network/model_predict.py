from ast import increment_lineno
import matplotlib
from numpy.core.fromnumeric import resize
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import activations
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
#模型加载
json_file=open('model_saved\model_json.json')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)
model.load_weights('model_saved\model_weight.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['acc'])

# (train_image,train_label),(test_image,test_label)=tf.keras.datasets.fashion_mnist.load_data()
# train_image=train_image/255.0  #归一化
# test_image=test_image/255.0

# model.evaluate(test_image,test_label)
# predict=model.predict(test_image)
# print(np.argmax(predict[0]))

# plt.imshow(test_image[0])
# plt.show()

label={0:'babaozhou',1:'bingtanghulu',2:'cookie',3:'icecream'}

def show_result(result):
    idx=np.argmax(result)
    return label[idx]

image_path=r'D:\python_work\Garbage_Classification\neural_network\temp\tingbaobing.jpg'
img_clssified=image.load_img(image_path,target_size=(256,256))
img_clssified=image.img_to_array(img_clssified)
img_clssified=np.expand_dims(img_clssified,axis=0)
result=model.predict(img_clssified)
print(result)
print("该垃圾类型为："+ show_result(result))


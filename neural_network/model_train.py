from ast import increment_lineno
import matplotlib
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import activations


(train_image,train_label),(test_image,test_label)=tf.keras.datasets.fashion_mnist.load_data()
train_image=train_image/255.0  #归一化
test_image=test_image/255.0
#plt.imshow(test_image[0])
#plt.show()

model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #多维数据扁平化，变成一维  28*28是图片尺寸
model.add(tf.keras.layers.Dense(128,activation='relu')) #隐藏层，128个隐藏单元  dense层单元太少会丢失有用信息，太多会过拟合
model.add(tf.keras.layers.Dropout(0.5)) #添加dropout层，抑制过拟合
model.add(tf.keras.layers.Dense(128,activation='relu')) #隐藏层，128个隐藏单元  dense层单元太少会丢失有用信息，太多会过拟合
model.add(tf.keras.layers.Dropout(0.5)) #添加dropout层，抑制过拟合
model.add(tf.keras.layers.Dense(128,activation='relu')) #隐藏层，128个隐藏单元  dense层单元太少会丢失有用信息，太多会过拟合
model.add(tf.keras.layers.Dropout(0.5)) #添加dropout层，抑制过拟合
model.add(tf.keras.layers.Dense(10,activation='softmax'))#输出层 10个单元，10个输出对应10中label的概率
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['acc'])
history=model.fit(train_image,train_label,epochs=10,validation_data=(test_image,test_label)) 

#训练过程可视化
# print(history.history.keys())
# plt.plot(history.epoch,history.history.get('loss'),label='loss')
# plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
# plt.legend()
# plt.show() 

#训练模型评估
#model.evaluate(test_image,test_label)
#predict=model.predict(test_image)
#print(np.argmax(predict[0]))

#model saved
model_json=model.to_json()
with open('model_saved\model_json.json','w') as json_file:
    json_file.write(model_json)
model.save_weights('model_saved\model_weight.h5')
model.save('model_saved\model.h5')
print('model saved.')
from ast import increment_lineno
import matplotlib
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import activations
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_path=r'D:\python_work\Garbage_Classification\neural_network\image_after_resize'
img_width,img_height = 256,256
batch_size = 16
cell=128
label_num=4

train_datagen = ImageDataGenerator(
       rescale=1./255,          #归一化
       shear_range=0.2,         #错切变换角度
       horizontal_flip=True,    #水平翻转
       validation_split=0.1)    #用作验证集的比例

test_datagen = ImageDataGenerator(
       rescale=1. / 255,
       validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
       file_path,
       target_size=(img_width, img_height),
       batch_size=batch_size,
       shuffle=True,
       class_mode="categorical", #对类型进行热编码："categorical",返回one-hot 编码标签
       save_format="jpeg",
       seed=0
       )

print(train_generator.labels.shape)



model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(img_width,img_height,3))) #多维数据扁平化，变成一维  
model.add(tf.keras.layers.Dense(cell,activation='relu')) #隐藏层，  dense层单元太少会丢失有用信息，太多会过拟合
#model.add(tf.keras.layers.Dropout(0.5)) #添加dropout层，抑制过拟合
model.add(tf.keras.layers.Dense(cell,activation='relu')) #隐藏层，  dense层单元太少会丢失有用信息，太多会过拟合
#model.add(tf.keras.layers.Dropout(0.5)) #添加dropout层，抑制过拟合
model.add(tf.keras.layers.Dense(cell,activation='relu')) #隐藏层，  dense层单元太少会丢失有用信息，太多会过拟合
#model.add(tf.keras.layers.Dropout(0.5)) #添加dropout层，抑制过拟合
model.add(tf.keras.layers.Dense(label_num,activation='softmax'))#输出层 10个单元，10个输出对应10中label的概率
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['acc'])
print("model training...")
history=model.fit(train_generator,epochs=10) 

#model saved
model_path=r'D:\python_work\Garbage_Classification\neural_network\model_saved\%s'
model_json=model.to_json()
with open(model_path %"model_json.json",'w') as json_file:
    print("model saving...")
    json_file.write(model_json)
model.save_weights(model_path %"model_weight.h4")
model.save(model_path %"model.h4")
print('model saved.')


#训练过程可视化
print(history.history.keys())
plt.plot(history.epoch,history.history.get('loss'),label='loss')
#plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
plt.legend()
plt.show() 

#训练模型评估
#model.evaluate(test_image,test_label)
#predict=model.predict(test_image)
#print(np.argmax(predict[0]))


from ast import increment_lineno
import matplotlib
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import activations
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D,Dense

#主要调整参数
train_file_path=r'D:\python_work\Garbage_Classification\neural_network\dataset\image_train'
validation_file_path=r'D:\python_work\Garbage_Classification\neural_network\dataset\image_val'
img_width,img_height = 256,256
batch_size = 15
cell=128
label_num=5
train_epochs=10

train_datagen = ImageDataGenerator(
       rescale=1./255,          #归一化
       shear_range=0.2,         #错切变换角度
       horizontal_flip=True,    #水平翻转
       validation_split=0.1)    #用作验证集的比例

val_datagen = ImageDataGenerator(
       rescale=1./255,
       validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
       train_file_path,
       target_size=(img_width, img_height),
       batch_size=batch_size,
       shuffle=True,
       class_mode="categorical", #对类型进行热编码："categorical",返回one-hot 编码标签
       save_format="jpeg"
       )

val_generator=val_datagen.flow_from_directory(
       validation_file_path,
       target_size=(img_width,img_height), 
       batch_size=batch_size,
       class_mode='categorical',
       save_format="jpeg"
)

model=tf.keras.Sequential()
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=(256,256,3))) #多维数据扁平化，变成一维  
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(cell,activation='relu'))
model.add(Dense(label_num,activation='softmax'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['acc'])
print("model training...")
history=model.fit(train_generator,epochs=train_epochs,validation_data=val_generator) 

#model saved
model_path=r'D:\python_work\Garbage_Classification\neural_network\model_saved\%s'
model_json=model.to_json()
with open(model_path %"model_json.json",'w') as json_file:
    print("model saving...")
    json_file.write(model_json)
model.save_weights(model_path %"model_weight.h5")
model.save(model_path %"model.h5")
print('model saved.')


#训练过程可视化
print(history.history.keys())
plt.plot(history.epoch,history.history.get('loss'),label='loss')
plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
plt.title('loss')
plt.legend()
plt.show() 

plt.plot(history.epoch,history.history.get('acc'),label='acc')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.title('accuracy')
plt.legend()
plt.show()

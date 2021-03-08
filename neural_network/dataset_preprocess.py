import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_path=r'D:\python_work\Garbage_Classification\neural_network\image_after_resize'

img_width,img_height = 256,256

batch_size = 16

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
       class_mode="categorical", #对类型进行热编码："categorical",返回one-hot 编码标签
       save_format="jpeg"
       )
print(len(train_generator.labels))
test_generator = test_datagen.flow_from_directory(
       file_path,
       target_size=(img_width, img_height),
       batch_size=batch_size,
       class_mode="categorical",
       save_format="jpeg"
       )


# def read_image(filename,label):
#     image_string=tf.io.read_file(filename)
#     image_resized=tf.image.resize_images(image_string,[256,256])
#     return image_resized,label

# file_path=r'D:\python_work\Garbage_Classification\neural_network\image_after_resize\kitchen waste_babaozhou'
# feature=[os.path.join(file_path,i) for i in os.listdir(file_path)]
# label=[0]*len(feature)
# dataset=tf.data.Dataset.from_tensor_slices((feature,label))
# dataset=dataset.map(read_image)
# print(dataset[0])
# dataset=dataset.batch(40)
# dataset=dataset.repeat(1)


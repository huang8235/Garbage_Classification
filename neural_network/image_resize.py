#发现tensorflow自带的ImageDataGenerator可以在读取图片的同时resize,故此脚本废弃不用

import cv2
import numpy as np
import os

def image_preporcess(image, target_size):

    # resize 尺寸
    ih, iw = target_size
    # 原始图片尺寸
    h,  w, _ = image.shape

    # 计算缩放后图片尺寸
    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    # 创建一张画布，画布的尺寸就是目标尺寸
    # fill_value=120为灰色画布
    image_paded = np.full(shape=[ih, iw, 3], fill_value=120)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2

    # 将缩放后的图片放在画布中央
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    # 归一化处理
    image_paded = image_paded / 255.

    return image_paded

if __name__=="__main__":
    folder_processed="kitchen waste_icecream"
    image_path = r"D:\python_work\Grabage_Classification\neural_network\temp\%s" %folder_processed    #加上r防止\后字母识别成转义字符
    save_path = r"D:\python_work\Grabage_Classification\neural_network\image_after_resize\%s\%s" 

    #image = cv2.imread(image_path)
    #img=image_preporcess(image,(256,256))

    #cv2.namedWindow("target_size_img",cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("target_size_img",img)
    #cv2.imwrite(r"D:\python_work\Grabage_Classification\neural_network\image_after_resize\after_image.jpeg",img*255.)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    f=os.listdir(image_path)
    for file in f:
        position = image_path+'\\'+ file
        image=cv2.imread(position)
        img=image_preporcess(image,(256,256))
        cv2.imwrite(save_path %(folder_processed,file),img*255.)
    print("resize successfully!")
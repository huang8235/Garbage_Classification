import os

path=r'D:\BaiduNetdiskDownload\垃圾图片库\厨余垃圾\厨余垃圾_饼干/'
english_name="cookie"
after_name='image_%s_' %english_name

f=os.listdir(path)
n=0
for i in f:
    oldname=path+f[n]
    newname=path+after_name+str(n+1)+'.jpeg'
    os.rename(oldname,newname)
    n=n+1
print("rename successfully!")

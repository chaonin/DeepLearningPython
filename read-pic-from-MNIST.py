# chaonin 24.5.14 from 51CTO
import gzip
import pickle
import pylab

# 以二进制只读格式读取图片及索引文件
with gzip.open('mnist.pkl.gz', 'rb') as f:
    # f.seek(0)
    img = pickle.load(f, encoding='latin1')

i = 0
j = 2
# 输出第j张图片
# i={0,1,2},i=0时为50000个数据的训练集
# i=1时为10000个数据的验证集
# i=2时为10000个数据的测试集
img_x = img[i][0][j].reshape(28, 28)
img_id = img[i][1][j]

print(img_id)
pylab.imshow(img_x)
pylab.gray()
pylab.show()

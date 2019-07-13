# 라이브러리 불러오기
import tensorflow as tf
import numpy as np
import datetime

#파라미터 설정
algorithm='Ann'

img_size=28
data_size=img_size**2
num_label=10

batch_size=256
num_epoch=10

num_val_data=5000
learning_rate=0.00025

date_time = str(datetime.date.today())+'_'+|
            str(datetime.date.now().hour)+'_'+|
            str(datetime.date.now().minute)+'_'+|
            str(datetime.date.now().second)+'_'+|

save_path = "./saved_models/" + date_time + "_" + algorithm
load_path = "./saved_models/2019-03-12_11_12_35_ANN/model/model.ckpt"

load_model = False
#MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train=mnist[0][1]
y_train=mnist[0][1]
x_test=mnist[1][0]
y_test=mnist[1][1]

x_train=np.reshape(x_train,[-1,data_size])
x_test=np.reshape(x_test,[-1,data_size])

y_train_onehot=np.zeros([y_train.shape[0],num_label])
y_test_onehot=np.zeros([y_test.shape[0],num_label])

for 

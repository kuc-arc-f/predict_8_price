# coding: utf-8
# 2018/12/30 17:27 : DL, 家賃の予測問題、層数の変更
#

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import time

from optimizer import SGD, Adam
from deep_price_net import DeepPriceNet

#
# 学習データ
global_start_time = time.time()
wdata = pd.read_csv("data.csv" )
wdata.columns =["no", "price","siki_price", "rei_price" ,"menseki" ,"nensu" ,"toho" ,"madori" ,"houi" ,"kouzou" ]
#print(wdata.head() )
#quit()

# conv=> num
sub_data = wdata[[ "no","price","siki_price", "rei_price" ,"menseki" ,"nensu" ,"toho" ] ]
sub_data = sub_data.assign(price=pd.to_numeric( sub_data.price))
print( sub_data.head() )
print(sub_data["price"][: 10])

# 説明変数に "price" 以外を利用
X = sub_data.drop("price", axis=1)
X = X.drop("no", axis=1)

#num_max_x= 10
num_max_x= 1000
X = (X / num_max_x )
print(X.head() )
print(X.shape )
#print( type( X) )
#print(X[: 10 ] )

# 目的変数
num_max_y= num_max_x
Y = sub_data["price"]
Y = Y / num_max_y
print(Y.max() )
print(Y.min() )
#quit()

# 学習データとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25 ,random_state=0)
#x_train_sub =x_train
#x_test_sub  =x_test
#x_train = x_train["x_dat"]
#x_test = x_test["x_dat"]
#print(type(x_train) )
#quit()
x_train =np.array(x_train, dtype = np.float32).reshape(len(x_train), 5)
y_train =np.array(y_train, dtype = np.float32).reshape(len(y_train), 1)
x_test  =np.array(x_test, dtype  = np.float32).reshape(len(x_test), 5 )
y_test =np.array(y_test, dtype   = np.float32).reshape(len(y_test), 1)

print( x_train.shape , y_train.shape  )
print( x_test.shape  , y_test.shape  )
#print(x_train[: 10])
#print(type(x_train ))
#quit()

# train
#max_epochs = 50
train_size = x_train.shape[ 0]
batch_size = 100
learning_rate = 0.01
iters_num = 10 * 1000   # 繰り返しの回数を適宜設定する    
#iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch = 500
#print(iter_per_epoch )
weight_init_std="relu"
#weight_init_std="sigmoid"
#quit()

# train
global_start_time = time.time()
network = DeepPriceNet(input_size=5
                                , hidden_size_list= [100, 100, 100, 100, 100] , output_size=1, 
                                weight_init_std=weight_init_std, use_batchnorm=True)
optimizer = SGD(lr=learning_rate)
#quit()

bn_train_acc_list = []   
train_loss_list = []     
epoch_cnt = 0

for i in range(iters_num):        
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    #
    grads = network.gradient(x_batch, y_batch)
    optimizer.update( network.params, grads)
    #
    if i % iter_per_epoch == 0:
        loss =network.loss(x_train , y_train, train_flg=True)
        train_loss_list.append(loss)
#        print("epoch:" + str(epoch_cnt) + ", time= " + str(time.time() - global_start_time) )
        print("epoch:" + str(epoch_cnt) + ", loss= " + str(loss ) )
        epoch_cnt += 1
# pred
pred =network.predict(x_test )
print(pred[: 10] * num_max_y )
print ('time : ', time.time() - global_start_time)
#save 
# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")
network.save_layers("p_layers.pkl")
print("Saved Network ,layers!")

#plt
a1=np.arange(len(train_loss_list) )
plt.plot(a1 , train_loss_list , label = "predict")
plt.legend()
plt.grid(True)
plt.title("loss price")
plt.xlabel("x")
plt.ylabel("loss")
plt.show()

quit()                                



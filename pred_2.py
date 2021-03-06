# -*- coding: utf-8 -*-
# 評価
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

# 学習データ
global_start_time = time.time()

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
#quit()
x_train =np.array(x_train, dtype = np.float32).reshape(len(x_train), 5)
y_train =np.array(y_train, dtype = np.float32).reshape(len(y_train), 1)
x_test  =np.array(x_test, dtype  = np.float32).reshape(len(x_test), 5 )
y_test =np.array(y_test, dtype   = np.float32).reshape(len(y_test), 1)

print( x_train.shape , y_train.shape  )
print( x_test.shape  , y_test.shape  )

#load model
weight_init_std="relu"
network = DeepPriceNet(input_size=5
                                , hidden_size_list=[10, 10, 10, 10, 10 ] , output_size=1, 
                                weight_init_std=weight_init_std, use_batchnorm=True)
network.load_params("params.pkl" )
network.load_layers("p_layers.pkl")
print("Load Network ,layers!")

#pred
y_val = network.predict( x_test )
y_val = y_val * num_max_y
#print(pred.shape )
#print(y_val[: 10] )
a1=np.arange(len(y_val) )
plt.plot(a1 , y_test *num_max_y , label = "y_test")
plt.plot(a1 , y_val , label = "predict")
plt.legend()
plt.grid(True)
plt.title("price predict")
plt.xlabel("x")
plt.ylabel("price")
plt.show()


from pandas import DataFrame
from pandas import Series
from pandas import concat
import keras
import xlrd
import os
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import LSTM
from keras.models import load_model
from math import sqrt
# import matplotlib.pyplot as plt
import numpy
import keras.backend as K


def Prepare_Excel_Data():

    d = []
    ans2d = []
    for dirname, dirnames, filenames in os.walk('contestdata50'):

        for filename in filenames:
            print("讀取檔案 : ",filename)
            if (filename == '.DS_Store'):
                continue
            myWorkbook = xlrd.open_workbook('contestdata50/' + filename)
            mySheets = myWorkbook.sheets()
            mySheet = mySheets[0]
            temparray = []
            thisexcelresult = 0
            for row in range(mySheet.nrows):
                for col in range(mySheet.ncols):
                    if (str(mySheet.cell(row, col).value).startswith('加工品質量測結果:')):
                        thisexcelresult = mySheet.cell(row, col).value.replace('加工品質量測結果:', '')



            for row in range(mySheet.nrows):
                if(row ==0):
                    continue
                target = []
                for col in range(mySheet.ncols):
                    target.append(thisexcelresult)
                    break

            for row in range(mySheet.nrows):
                values = []
                result = []
                check = True
                for col in range(mySheet.ncols):
                    if (str(mySheet.cell(row, col).value).startswith('加工品質量測結果:')):
                        check = False
                        result.append(str(mySheet.cell(row, col).value).replace('加工品質量測結果:',''))
                        ans2d.append(result)
                    values.append(mySheet.cell(row, col).value)
                if check:
                    temparray.append(values)
            d.append(temparray)

    return d,ans2d

def array_to_numpy(d,ans2d):
    d = numpy.array(d)
    ans2d = numpy.array(ans2d).astype("float32")

    return d,ans2d



def zero_mean(value):
    value_zeromean = (value - value.mean(axis=(0, 1))) / value.std(axis=(0, 1))
    return value_zeromean


def negativeOne_to_One_2dim(value):
    value_min = value.min(axis=(0),keepdims=True)
    value_max = value.max(axis=(0),keepdims=True)
    Scaled_value = ((value-value_min)/(value_max-value_min))*2-1

    return Scaled_value

def root_mean_squared_error(y_true, y_pred): #loss函式
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def reverse_NegativeOne_to_One(prediction_value,real_value):
    real_value_min = real_value.min(axis=0)
    real_value_max = real_value.max(axis=0)


    value = (((prediction_value+1)/2)*(real_value_max-real_value_min))+real_value_min
    return value


d,ans2d = Prepare_Excel_Data()

d,ans2d = array_to_numpy(d,ans2d)

Data_Zeromean = zero_mean(d)

Ans2d_NegativeOne_to_One = negativeOne_to_One_2dim(ans2d) #ans2d為原domain答案

X_test,y_test = Data_Zeromean[-10:,:,:],Ans2d_NegativeOne_to_One[:,:] #測試資料、答案取全部 (輸入取原始資料直接轉zscore，答案取介於-1,1)      #答案沒有轉zscore 只有輸入轉zscore


import keras.losses
import sys
keras.losses.root_mean_squared_error = root_mean_squared_error


if sys.argv[1] == 'CPU':
    modelp = load_model('CPUversion_Model')
elif sys.argv[1] == 'GPU':
    modelp = load_model('GPUversion_Model')
else:
    print('Use CPU model to predict')
    modelp = load_model('CPUversion_Model')
modelp.compile(loss=root_mean_squared_error, optimizer='adam')
predictions = modelp.predict(X_test,batch_size=1)

x_axis = numpy.linspace(1,10,10)
x_axis_for_test = numpy.linspace(1,40,40)

predictions = predictions.ravel()
y_test = y_test.ravel()
# plt.plot(x_axis,predictions,label='predictions')
# plt.plot(x_axis_for_test,y_test,label='realvalue')
# plt.legend()
# plt.show()

ans2d = ans2d.ravel()

predictions_rev = reverse_NegativeOne_to_One(prediction_value=predictions,real_value=ans2d)


# plt.plot(x_axis,predictions_rev,label='predictions')
# plt.plot(x_axis_for_test,ans2d,label='realvalue')
# plt.legend()
# plt.show()

print(predictions_rev[:])

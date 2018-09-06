import keras
import xlrd
from keras.callbacks import Callback,EarlyStopping
import os
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,CuDNNLSTM,LSTM
from keras.models import load_model
import numpy
import keras.backend as K
import sys

def Prepare_Excel_Data():

    d = []
    ans = []
    ans2d = []
    for dirname, dirnames, filenames in os.walk('contestdata40'):

        for filename in filenames:
            print("讀取檔案 : ",filename)
            if (filename == '.DS_Store'):
                continue
            myWorkbook = xlrd.open_workbook('contestdata40/' + filename)
            mySheets = myWorkbook.sheets()
            mySheet = mySheets[0]
            temparray = []
            tempans = []
            check = True
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

                tempans.append(target)
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
            ans.append(tempans)

    return d,ans2d

def array_to_numpy(d,ans2d):
    d = numpy.array(d)
    ans2d = numpy.array(ans2d).astype("float32")

    return d,ans2d



def zero_mean(value):
    value_zeromean = (value - value.mean(axis=(0, 1))) / value.std(axis=(0, 1))
    return value_zeromean

def negativeOne_to_One_3dim(value):
    value_min = value.min(axis=(0,1),keepdims=True)
    value_max = value.max(axis=(0,1),keepdims=True)
    Scaled_value = ((value-value_min)/(value_max-value_min))*2-1

    return Scaled_value

def negativeOne_to_One_2dim(value):
    value_min = value.min(axis=(0),keepdims=True)
    value_max = value.max(axis=(0),keepdims=True)
    Scaled_value = ((value-value_min)/(value_max-value_min))*2-1

    return Scaled_value




def root_mean_squared_error(y_true, y_pred): #loss函式
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))




#====================================================================model structure
def BuildModel(lr):

    model = Sequential()
    model.add(CuDNNLSTM(64, batch_input_shape=(1, 7500, 4), return_sequences=True, stateful=True))
    model.add(Dropout(0.5))
    model.add(CuDNNLSTM(128, batch_input_shape=(1, 7500, 4), return_sequences=True, stateful=True))
    model.add(Dropout(0.5))
    model.add(CuDNNLSTM(256, batch_input_shape=(1, 7500, 4), return_sequences=True, stateful=True))
    model.add(Dropout(0.5))
    model.add(CuDNNLSTM(128, batch_input_shape=(1, 7500, 4), return_sequences=True, stateful=True))
    model.add(Dropout(0.5))
    model.add(CuDNNLSTM(64, batch_input_shape=(1, 7500, 4), return_sequences=False, stateful=True))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='tanh'))
    adam = keras.optimizers.Adam(lr=lr)

    model.compile(loss=root_mean_squared_error, optimizer=adam)

    return model
#====================================================================

#====================================================================model structure
def BuildModel_CPU(lr):

    model = Sequential()
    model.add(LSTM(64, batch_input_shape=(1, 7500, 4), return_sequences=True, stateful=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, batch_input_shape=(1, 7500, 4), return_sequences=True, stateful=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256, batch_input_shape=(1, 7500, 4), return_sequences=True, stateful=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, batch_input_shape=(1, 7500, 4), return_sequences=True, stateful=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, batch_input_shape=(1, 7500, 4), return_sequences=False, stateful=True))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='tanh'))
    adam = keras.optimizers.Adam(lr=lr)

    model.compile(loss=root_mean_squared_error, optimizer=adam)

    return model
#====================================================================

# def Curve_of_training(losstraining,epochs):

    # losstraing = numpy.array(losstraining).astype('float32')
    # x = numpy.linspace(1,epochs,epochs)
    # plt.title('contestN1 plt')
    # plt.plot(x,losstraing,label='loss')
    # plt.plot(x,lossvalidation,label='val_loss')
    # import datetime
    # plt.savefig('fig_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    # plt.show()

#==================================================================== training

Minimumloss = 0
ThatEpochs = 0

def trainingModel(training_data,training_label,epochs,lr,save_model_name,save_weight_name):

    if sys.argv[1] == 'CPU':
        model = BuildModel_CPU(lr)
    elif sys.argv[1] == 'GPU' :
        model = BuildModel(lr)
    else:
        model = BuildModel_CPU(lr)

    losstraing = []  #紀錄每個ep的loss

    for i in range(epochs):
        print('epoch : ',i+1,'    =================   ')
        history = model.fit(training_data, training_label, epochs=1, batch_size=1, verbose=1, shuffle=False)

        if i ==0:
            model.save(save_model_name)
            model.save_weights(save_weight_name)
            pass
        elif i>0:
         if history.history['loss'][0] < min(losstraing):
            global Minimumloss
            global ThatEpochs
            Minimumloss= history.history['loss']
            ThatEpochs = i+1
            print(history.history['loss'],'\t儲存model\t目前最低loss : ',min(losstraing))
            model.save(save_model_name)
            model.save_weights(save_weight_name)
         else:
            print('不儲存model','\t目前最低loss : ',min(losstraing))

        losstraing.append(history.history['loss'][0])

        model.reset_states()

    # Curve_of_training(losstraing,epochs)



#==================================================================== 載入model的predict 印圖


def reverse_NegativeOne_to_One(prediction_value,real_value):
    real_value_min = real_value.min(axis=0)
    real_value_max = real_value.max(axis=0)


    value = (((prediction_value+1)/2)*(real_value_max-real_value_min))+real_value_min
    return value



# def Curve_of_Prediction(Minimumloss,predictions,real_value_NegativeOne_to_One,real_value_original_domain,data_amount,epochs,lr):
#
#     x_axis = numpy.linspace(1, data_amount, data_amount)
#
#     predictions = predictions.ravel()
#     real_value_original_domain = real_value_original_domain.ravel()
#
#
#     reverse_predictions = reverse_NegativeOne_to_One(prediction_value=predictions,real_value=real_value_original_domain)
#
#     real_value_NegativeOne_to_One = real_value_NegativeOne_to_One.ravel()
#
#     error_of_P_and_R = real_value_NegativeOne_to_One-predictions
#
#     RMSE_All_NegativeOne_to_One = numpy.sqrt(numpy.mean((predictions[:]-real_value_NegativeOne_to_One[:])**2))
#     RMSE_TOP30_NegativeOne_to_One = numpy.sqrt(numpy.mean((predictions[:30]-real_value_NegativeOne_to_One[:30])**2))
#     RMSE_LAST10_NegativeOne_to_One = numpy.sqrt(numpy.mean((predictions[-10:]-real_value_NegativeOne_to_One[-10:])**2))
#
#
#
#     RMSE_All_Original_Domain = numpy.sqrt(numpy.mean((reverse_predictions[:]-real_value_original_domain[:])**2))
#     RMSE_TOP30_Original_Domain = numpy.sqrt(numpy.mean((reverse_predictions[:30]-real_value_original_domain[:30])**2))
#     RMSE_LAST10_Original_Domain = numpy.sqrt(numpy.mean((reverse_predictions[-10:]-real_value_original_domain[-10:])**2))
#
#     plt.figure(1,figsize=(20,10))
#     sub1 = plt.subplot(221)
#     sub1.set_title("Epochs : "+str(epochs) + "     LR : "+str(lr) + "\n Minimumloss : " + str(Minimumloss) +"  SavedEpochs : "+str(ThatEpochs))
#     plt.plot(x_axis, predictions, label='predictions')
#     plt.plot(x_axis, real_value_NegativeOne_to_One, label='real_value_of_-1_to_1')
#     plt.legend()
#
#     sub2 = plt.subplot(222)
#     sub2.set_title('RMSE_All : ' + str(RMSE_All_NegativeOne_to_One) + "\n" + "RMSE_TOP30 : " + str(RMSE_TOP30_NegativeOne_to_One) + "    " + "RMSE_LAST10 : " + str(RMSE_LAST10_NegativeOne_to_One))
#     plt.ylim(-1, 1)
#     plt.plot(x_axis, error_of_P_and_R, label='error_of_-1_to_1')
#     plt.legend()
#
#     plt.subplot(223)
#     plt.plot(x_axis, reverse_predictions, label='reversed_prediction')
#     plt.plot(x_axis, real_value_original_domain, label='real_value_original_domain')
#     plt.legend()
#
#     error_of_P_and_R_origin_domain = real_value_original_domain - reverse_predictions
#
#     sub4 = plt.subplot(224)
#     sub4.set_title("RMSE_All : "+str(RMSE_All_Original_Domain)+"\n"+"RMSE_TOP30 : "+str(RMSE_TOP30_Original_Domain)+"    RMSE_LAST10 : "+str(RMSE_LAST10_Original_Domain))
#     plt.ylim(-1, 1)
#     plt.plot(x_axis, error_of_P_and_R_origin_domain, label='error_of_original_domain')
#     plt.legend()
#     import datetime
#     plt.savefig('fig_'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
#     plt.show()


def Predict_model(Minimumloss,testing_data,real_value_NegativeOne_to_One,real_value_original_domain,model_name,data_amount,epochs,lr):
    model = load_model(model_name,custom_objects={ 'root_mean_squared_error':root_mean_squared_error })
    model.compile(loss=root_mean_squared_error, optimizer='adam')
    predictions = model.predict(testing_data,batch_size=1)
    # Curve_of_Prediction(Minimumloss=Minimumloss,predictions=predictions,real_value_NegativeOne_to_One=real_value_NegativeOne_to_One,real_value_original_domain=real_value_original_domain,data_amount=data_amount,epochs=epochs,lr=lr)






d,ans2d = Prepare_Excel_Data()

d,ans2d = array_to_numpy(d,ans2d)

Data_Zeromean = zero_mean(d)

Ans2d_NegativeOne_to_One = negativeOne_to_One_2dim(ans2d) #ans2d為原domain答案

X_train,y_train = Data_Zeromean[:,:,:],Ans2d_NegativeOne_to_One[:,:] #訓練資料、答案取前30筆 (輸入取原始資料直接轉zscore，答案取介於-1,1)      #答案沒有轉zscore 只有輸入轉zscore
X_test,y_test = Data_Zeromean[:,:,:],Ans2d_NegativeOne_to_One[:,:] #測試資料、答案取全部 (輸入取原始資料直接轉zscore，答案取介於-1,1)      #答案沒有轉zscore 只有輸入轉zscore



epochs = 3
lr = 0.0001



trainingModel(training_data=X_train,training_label=y_train,epochs=epochs,lr=lr,save_model_name='firsttryN1_loss',save_weight_name='firsttryN1_weight_loss')


Predict_model(Minimumloss=Minimumloss,testing_data=X_test,real_value_NegativeOne_to_One=Ans2d_NegativeOne_to_One,real_value_original_domain=ans2d,model_name='firsttryN1_loss',data_amount=40,epochs=epochs,lr=lr)










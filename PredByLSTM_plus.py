#coding:utf-8
import pandas as pd
import time
import os
import numpy as np
from keras.utils import to_categorical, plot_model
from keras.layers import Dense, LSTM, Activation, Input, Dropout
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pylab as plt


###
# from poi input(x) to next poi to next cq cu
###


class CLIENT_STATE:
    def __init__(self):
        self.num_poi_class = 8
        self.num_sl_class = 7
        self.train_epoch = 200
        self.look_back = 6
        self.Batch_Size = 5
        self.lstm_units = 50

    def create_lb_dataset(self, data_x, data_y, l_back):
        dataX, dataY, dataYY = [], [], []
        for i in range(len(data_x) - l_back):
            a = data_x[i:(i + l_back)]
            dataX.append(a)
            dataY.append(data_y[i + l_back - 1])
        return np.array(dataX), np.array(dataY)

    def create_raw_data(self, value, split, is_train=True):
        raw_x = []
        raw_y = []
        raw_yy = []

        if is_train:

            # generate train data
            for i in range(split):

                # 工作日或调休
                if value[i][0] == 0:  # daytype,whatday,poi,dt,dt_scale
                    raw_x.append([value[i][2], value[i][1], value[i][4]])
                    raw_y.append(value[i + 1][2])
        else:

            # generate test data
            for i in range(split, len(value) - 1):  # len(value)-1

                # 工作日或调休
                if value[i][0] == 0 or value[i][0] == 2:
                    # print(value[i][0], sn_scale, en_scale)
                    raw_x.append([value[i][2], value[i][1], value[i][4]])
                    raw_y.append(value[i + 1][2])

        raw_x = np.array(raw_x)
        raw_y = np.array(raw_y)
        raw_y_onehot = to_categorical(raw_y, num_classes=self.num_poi_class)

        return raw_x, raw_y_onehot

    def lstm_model(self, x_train, y_train, x_test, y_test):

        # input model
        inputs_01 = Input(shape=(self.look_back, x_train.shape[2]), name='input_layer')

        lstm = LSTM(units=self.lstm_units, name='lstm_layer')(inputs_01)
        dropout = Dropout(0.5)(lstm)
        dense_01 = Dense(32, activation='relu', name='dense_01')(dropout)
        dense_02 = Dense(16, activation='relu', name='dense_02')(dense_01)

        # output poi
        output_poi = Dense(self.num_poi_class, activation='softmax', name='output_poi')(dense_02)

        model = Model(inputs=[inputs_01], outputs=[output_poi])

        # plot model
        # plot_model(model, to_file='./pred_model/model.png', show_shapes=True)
        print(model.summary())

        adam = Adam(decay=0.000001)

        model.compile(optimizer=adam,
                      loss={'output_poi': 'categorical_crossentropy'},
                      loss_weights={'output_poi': 1},
                      metrics=['accuracy'],
                      )

        history = model.fit({'input_layer': x_train},
                            {'output_poi': y_train},
                            epochs=self.train_epoch,
                            batch_size=self.Batch_Size,
                            verbose=True,
                            shuffle=True,
                            validation_data=({'input_layer': x_test}, {'output_poi': y_test})
                            )

        eval = model.evaluate({'input_layer': x_test}, {'output_poi': y_test})

        print(model.metrics_names)
        print(eval)
        print(type(eval))
        print('\ntotal loss: ', eval[0])
        print('poi loss: ', eval[1])
        print('poi acc: ', eval[2])

        y_pred = model.predict(x_test)
        pred_poi = [np.argmax(one_hot) for one_hot in y_pred[0]]
        real_poi = [np.argmax(one_hot) for one_hot in y_test]

        print('\nreal_poi: ')
        print(real_poi)
        print('pred_poi: ')
        print(pred_poi)


        tra_poi_loss = history.history['output_poi_loss']
        val_poi_loss = history.history['val_output_poi_loss']
        tra_poi_acc = history.history['output_poi_acc']
        val_poi_acc = history.history['val_output_poi_acc']

        # plot train history
        plt.subplot(2, 1, 1)
        plt.plot(tra_poi_loss, label='poi_loss')
        plt.plot(val_poi_loss, label='val_poi_loss')
        plt.legend()

        plot_title = 'poi_loss: ' + str(eval[1])[:6]
        plt.title(plot_title)
        plt.subplot(2, 1, 2)
        plt.plot(tra_poi_acc, label='poi_acc')
        plt.plot(val_poi_acc, label='val_poi_acc')
        plt.legend()
        plot_title = ' poi_acc: ' + str(eval[3])[:6]
        plt.title(plot_title)
        plt.show()

        tra_poi_loss = np.array(tra_poi_loss).reshape((1, len(tra_poi_loss)))  # reshape是为了能够跟别的信息组成矩阵一起存储
        val_poi_loss = np.array(val_poi_loss).reshape((1, len(val_poi_loss)))

        tra_poi_acc = np.array(tra_poi_acc).reshape((1, len(tra_poi_acc)))  # reshape是为了能够跟别的信息组成矩阵一起存储
        val_poi_acc = np.array(val_poi_acc).reshape((1, len(val_poi_acc)))

        np_out = np.concatenate([tra_poi_loss, val_poi_loss,
                                 tra_poi_acc, val_poi_acc], axis=0)

        np.savetxt('./lstm_acc_loss6.txt', np_out)

        # # save model and weights:
        # model_path = './pred_model/c1_pred_model2.h5'
        # model.save(model_path)
        # print('Saved trained model at %s ' % model_path)


if __name__ == "__main__":

    fc = CLIENT_STATE()

    dataset = pd.read_csv('./c1_poi_span.csv')

    value = dataset.values

    split = int(len(value) * 0.8)  # split % of values as training data

    look_back = 6

    train_X, train_y_onehot = fc.create_raw_data(value, split, is_train=True)
    test_X, test_y_onehot = fc.create_raw_data(value, split, is_train=False)

    x_train, y_train = fc.create_lb_dataset(train_X, train_y_onehot, look_back)
    x_test, y_test = fc.create_lb_dataset(test_X, test_y_onehot, look_back)


    # pred poi by lstm
    fc.lstm_model(x_train, y_train, x_test, y_test)

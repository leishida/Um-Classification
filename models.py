import keras
import numpy as np
from keras import backend as K
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, GlobalAveragePooling2D, Lambda
from keras.layers import Dropout, Flatten, Conv2D, add, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras import regularizers, optimizers, initializers

from keras import regularizers, losses, initializers
from keras.callbacks import Callback, LearningRateScheduler

from dataset import load_dataset
from loss import get_forward_loss
from keras.utils.np_utils import to_categorical

class BaseModel():
    def compile_model(self, model, base_mode, Pi, priors_corr, prior_test, mode):
        training_loss = get_forward_loss(priors_corr)
        model.compile(loss=training_loss, optimizer=self.optimizer)
        self.model = model
        self.base_mode = base_mode

    def fit_model(self, U_sets, x_train_total, batch_size, epochs, x_test, y_test, Pi, priors_corr, prior_test, mode):
        test_loss = TestLoss(self.base_mode, x_test, y_test, mode)
        X_train = x_train_total[U_sets[0],:]
        Y_train = np.zeros(len(X_train), dtype=np.int32).reshape(len(X_train), 1)
        for i in range(1, len(Pi)):
            X_train = np.concatenate((X_train, x_train_total[U_sets[i]]))
            Y_train = np.concatenate((Y_train, i * np.ones((len(U_sets[i]), 1))))
        print(X_train.shape)
        perm = np.random.permutation(len(Y_train))
        X_train, Y_train = X_train[perm], Y_train[perm]
        Y_train = to_categorical(Y_train)
        h = self.model.fit(X_train,
                  Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[test_loss])
        loss_test = test_loss.test_losses
        return h.history, loss_test


class MultiLayerPerceptron(BaseModel):
    def __init__(self, dataset, sets, set_sizes, Pi, mode,
                 weight_decay=1e-4):
        self.sets = sets
        self.set_sizes = set_sizes
        self.Pi = Pi
        self.mode = mode
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.optimizer = None

    def adaption_layer(self, g):
        c = 0
        d = 0
        output = []
        sets = len(self.Pi)
        for i in range(sets):
            c += self.priors_corr[i] * (self.Pi[i] - self.prior_test)
            d += (1 - self.Pi[i]) * self.priors_corr[i] * self.prior_test
        for i in range(sets):
            a = self.priors_corr[i] * (self.Pi[i] - self.prior_test)
            b = (1 - self.Pi[i]) * self.prior_test * self.priors_corr[i]
            output.append((a * g + b) / (c * g + d))
        res = K.concatenate(output, axis = 1)
        return res

    def build_model(self, priors_corr, prior_test, Pi, input_shape, mode):
        self.prior_test = prior_test
        self.priors_corr = priors_corr
        self.Pi = Pi
        input = Input(shape=input_shape)

        x = Dropout(0.2, input_shape = input_shape)(input)
        x = Dense(300, use_bias=False, input_shape=input_shape,
                  kernel_initializer=initializers.lecun_normal(seed=1),
                  kernel_regularizer=regularizers.l2(self.weight_decay))(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(300, use_bias=False,
                  kernel_initializer=initializers.lecun_normal(seed=1),
                  kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(300, use_bias=False,
                  kernel_initializer=initializers.lecun_normal(seed=1),
                  kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(300, use_bias=False,
                  kernel_initializer=initializers.lecun_normal(seed=1),
                  kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        

        g = Dense(1, use_bias=True,
              kernel_initializer=initializers.lecun_normal(seed=1),
              activation = 'sigmoid',
              name = "base_mode")(x)
        base_mode = Model(inputs=input, outputs=g)
        g_bar = Lambda(self.adaption_layer, name = "last_layer")(g)
        model = Model(inputs=input, outputs=g_bar)
            
        self.compile_model(model=model,
                  base_mode=base_mode,
                  Pi=self.Pi,
                  priors_corr = priors_corr,
                  prior_test = prior_test,
                  mode=self.mode,
                  )
        
# Test risk by 01-loss
class TestLoss(Callback):
    def __init__(self, base_mode, x_test, y_test, mode):
        self.base_mode = base_mode
        self.x_test = x_test
        self.y_test = y_test
        self.mode = mode

    def on_train_begin(self, logs={}):
        self.test_losses = []

    def on_epoch_end(self, epoch, logs={}):
        perm = np.random.permutation(len(self.x_test))
        self.x_test, self.y_test = self.x_test[perm], self.y_test[perm]
        nb_y_test = np.size(self.y_test)
        y_test_pred_base = self.base_mode.predict(self.x_test, batch_size=1000)
        y_test_pred = (y_test_pred_base >= 1/2) + 0
        test_loss = np.sum(np.not_equal(y_test_pred, self.y_test).astype(np.int32)) / nb_y_test
        self.test_losses.append(test_loss)

        print("\n Test loss: %f" % (test_loss))
        print("============================================================================")
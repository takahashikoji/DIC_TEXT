import keras
from keras.datasets import mnist

(x_train , y_train) , (x_test , y_test) = mnist.load_data()

img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]
num_classes = len(set(y_train))

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = x_train.shape[1:4]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

from keras.models import Sequential
from keras.layers import Dense , Flatten
from keras.layers import Conv2D , MaxPooling2D

model = Sequential()

from keras.layers import BatchNormalization

model.add(Conv2D(2,kernel_size=3,
                strides=1,
                padding = 'same',
                activation = 'relu',
                 bias = True ,
                input_shape = input_shape))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),
                      strides=(1,1),
                       padding = 'valid'))


model.add(Conv2D(2,kernel_size=3,
                strides=1,
                padding = 'same',
                activation = 'relu',
                 bias = True
                ))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),
                      strides=(1,1),padding = 'valid'))

from keras.layers import Dropout

model.add(Flatten())
model.add(Dense(210, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(210, activation='relu'))
model.add(Dense(num_classes , activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])

model.fit(x_train , y_train,
         batch_size=128,
         epochs = 3,
         verbose =1,
         validation_data=(x_test , y_test))
score = model.evaluate(x_test , y_test , verbose=1)
print('Test loss' , score[0])
print('test accuracy' , score[1])                       

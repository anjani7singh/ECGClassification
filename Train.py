from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
import keras
IMAGE_SIZE=[128,128]
model = Sequential()

model.add(Conv2D(64, (3,3),strides = (1,1), input_shape =IMAGE_SIZE+[3],kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides= (2,2)))

model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides= (2,2)))

model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides= (2,2)))

model.add(Flatten())

model.add(Dense(2048))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Loading Data
import zipfile
zip = zipfile.ZipFile('./gdrive/My Drive/ECGData.zip')
zip.extractall()
from keras.preprocessing.image import ImageDataGenerator
traindata_gen=ImageDataGenerator(rescale=1./255)
testdata_gen=ImageDataGenerator(rescale=1./255)
training_set = traindata_gen.flow_from_directory('./ECGData/Trainset',target_size=(128,128),batch_size=32,class_mode='categorical')
test_set = testdata_gen.flow_from_directory('./ECGData/Testset',target_size=(128,128),batch_size=32,class_mode='categorical')
model.fit_generator(training_set,steps_per_epoch=1000,epochs=25,validation_data=test_set,validation_steps=2000)

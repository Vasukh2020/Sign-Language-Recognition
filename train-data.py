import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
model = Sequential()
model.add(Convolution2D(64, (3, 3), input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=248, activation='relu'))
model.add(Dense(units=26, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
model.summary()
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=1,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=1,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
model.fit(
        training_set,
        steps_per_epoch=520, #no of images in training datase
        epochs=10,
        validation_data=test_set,
        validation_steps=130)#no of images in testing dataset
score=model.evaluate(training_set, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))
score=model.evaluate(test_set, verbose=0)
print("Test Accuracy: %.2f%%" % (score[1]*100))
model_json = model.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model-bw.h5')
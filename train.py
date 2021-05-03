from keras_efficientnets import EfficientNetB1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

#-------------------------static value
IMG_SIZE = 224
EPOCHS = 30

#-------------------------for tensor.op
img_augmentation = Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

#-------------------------create model
def create_model():
	inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
	x = img_augmentation(inputs)
	outputs = EfficientNetB1(
		weights=None,
		include_top=False,
		classes = 2 
		)(x)
	model = tf.keras.Model(inputs, outputs)
	model.compile(
		optimizer="sgd",
		loss="binary_crossentropy",
		metrics=["accuracy"]
		)
	model.summary()
	return model

#------------------------create model with pretrained weight
def pretrained_weight_model():
	inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
	x = img_augmentation(inputs)
	model = EfficientNetB1(include_top=False, input_tensor=x, weights="imagenet")
	model.trainable = False
	x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
	x = tf.keras.layers.BatchNormalization()(x)
	top_dropout_rate = 0.2
	x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
	outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred")(x)
	model = tf.keras.Model(inputs, outputs, name="EfficientNet")
	#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
	optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9,nesterov=True)
	model.compile(
		optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
	)
	model.summary()
	return model

#------------------------for checking training history
def plot_history(hist):
	plt.plot(hist.history["accuracy"])
	plt.plot(hist.history["val_accuracy"])
	plt.title("model accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend(["train", "validation"], loc="upper left")
	plt.show()

#------------------------training
def train():
	X_train, X_test, Y_train, Y_test = np.load('/your/path/img_data.npy', allow_pickle=True)
	X_train = np.append(X_train,X_test, axis=0)
	Y_train = np.append(Y_train,Y_test, axis=0)
	model = pretrained_weight_model() 
	epochs = EPOCHS
	hist = model.fit(X_train,Y_train, epochs=epochs, validation_split=0.1, verbose=2)
	plot_history(hist)
	model.save("/your/path/model1.h5")

#train()
model2 = tf.keras.models.load_model("/your/path/model1.h5")
img = cv2.imread("/your/path/.jpg")
img2 = cv2.imread("/your/path/.jpg")
img2 = np.expand_dims(img2, axis=0)
imgs = np.expand_dims(img, axis=0)
imgs = np.concatenate((imgs, img2), axis=0)

prediction = model2.predict(imgs)
label = ['oyster','scallop']

while(True):
	for i,img in enumerate(imgs):
		img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
		cv2.imshow('predict{}'.format(i),cv2.putText(img, label[np.argmax(prediction[i])], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2))
	key=cv2.waitKey(10)
	if key == 27:
		break


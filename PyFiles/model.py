from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.activations import softmax, sigmoid, relu
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive
from ten
keras.callbacks import ReduceLROnPlateau

drive.mount("/content/drive")

x_train = tf.convert_to_tensor(np.load("/content/drive/MyDrive/...."),dtype=np.float32)
y_train = np.load("/content/drive/MyDrive/....")

x_test = tf.convert_to_tensor(np.load("/content/drive/MyDrive/....."),dtype=np.float32)
y_test = np.load("/content/drive/MyDrive/....")

print(x_train.shape)
print(x_test.shape)


model = Sequential()

model.add(Input((x_train[0].shape)))
model.add(Conv2D(256,(3,3),padding="same",activation=relu)) #256,512,1024 -> decresed to the current filter and also added regulizers and callbacks
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(512,(3,3),padding="same",activation=relu))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(1024,(3,3),padding="same",activation=relu))
model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128,activation=relu,kernel_regularizer=l2(0.01)))
model.add(Dense(128,activation=relu,kernel_regularizer=l2(0.01)))
model.add(Dense(1,activation=sigmoid))

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )


model.compile(optimizer=Adam(0.001),
              loss='binary_crossentropy',
              metrics=["accuracy"],
              )
history = model.fit(x_train,y_train,batch_size=32,validation_split=0.2,epochs=100,callbacks=[reduce_lr]
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

test_loss, test_accuracy = model.evaluate(x_test, y_test,batch_size=32, verbose=2)

print(f'Test Loss: {test_loss}')

print(f'Test Accuracy: {test_accuracy}')
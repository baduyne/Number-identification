import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
import numpy as np

from input_instance import create_img  # Đảm bảo file này có hàm `create_img()`

def pre_pro(): 
    path = "mnist_data.npz"

    # Load dữ liệu từ file
    with np.load(path) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']

    # Reshape ảnh MNIST thành (28,28,1) để dùng CNN
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # One-hot encoding nhãn
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


# Xây dựng mô hình CNN
def convolutional_model(num_classes):
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Lớp đầu ra phải đúng số lớp

    # Compile mô hình
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def training(): 
   
    x_train, y_train, x_test, y_test = pre_pro()
    num_classes = y_test.shape[1]  # Xác định số lớp từ dữ liệu
    print(num_classes)
    # Xây dựng model với đúng số lớp
    model = convolutional_model(num_classes)

    # Huấn luyện mô hình
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)

    # Đánh giá mô hình
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: {:.2f}% \n Error: {:.2f}%".format(scores[1] * 100, (1 - scores[1]) * 100))
    return model

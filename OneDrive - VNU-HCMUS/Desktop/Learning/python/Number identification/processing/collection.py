import cv2 as cv 
import keras
import numpy as np 

# import data
from keras.datasets import mnist

 
def load_and_save_dataset(path):
    
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Lưu dữ liệu MNIST dưới dạng nén .npz
    np.savez_compressed(path, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)

    return path

# read data 
def read_dataset(path): 
    data = np.load(path)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    return x_train, y_train, x_test, y_test


# show image 
def show_images(train,target, size = 10): 
    i = 0 
    while True:  
        cv.imshow("image", train[i])
        i = i + 1   
        print(target[i])
        cv.waitKey(20)    
        
        if size == i : 
            break
        

def main():
    
    path = "mnist_data.npz"
    
    load_and_save_dataset(path)
    x_train, y_train, x_test, y_test = read_dataset(path)
    
    show_images(x_train,y_train, size = 20)
    
    
if __name__ == "__main__": 
    main()
    
    
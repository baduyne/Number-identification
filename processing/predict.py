from build_model import training
from input_instance import create_img
import numpy as np
import sys

def main(): 
    if len(sys.argv) != 2 :  # Kiểm tra nếu không có đúng 1 tham số
        print("Limit to one argument.")
        return

    iter = int(sys.argv[1])  # Lấy tham số từ dòng lệnh
    print(f"value: {iter}")
    
      
    cnn_model = training()

    for i in range(0, iter): 
        example = None
        example = create_img()
       
        example = example.reshape(1, 28, 28, 1)  # Thêm batch dimension
        example = example.astype("float32") / 255.0  # Chuẩn hóa về [0,1]

        # Dự đoán một ảnh
        predict = cnn_model.predict(example)

        predicted_label = np.argmax(predict)
        print(f"Nhận diện: {predicted_label}")


if __name__ == "__main__":
    main()
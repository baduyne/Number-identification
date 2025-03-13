import cv2 as cv
import numpy as np

# Ảnh đen gốc (sử dụng làm template)
BLACK_IMAGE = np.zeros((560, 560, 1), dtype=np.uint8)

# Hàm thu nhỏ ảnh
def rescale(frame, scale=0.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Biến kiểm soát trạng thái vẽ
drawing = False  

# Hàm xử lý sự kiện chuột
def draw_number(event, x, y, flags, param):
    global drawing, drawed_image

    if event == cv.EVENT_LBUTTONDOWN:  # Nhấn chuột để bắt đầu vẽ
        drawing = True

    elif event == cv.EVENT_MOUSEMOVE:  # Di chuyển chuột khi đang vẽ
        if drawing:
            cv.circle(drawed_image, (x, y), 15, 255, -1)  # Nét dày hơn để khi thu nhỏ không mất nét

    elif event == cv.EVENT_LBUTTONUP:  # Thả chuột để dừng vẽ
        drawing = False

def create_img():
    global drawed_image

    # Tạo bản sao từ ảnh đen gốc để không bị ảnh hưởng giữa các lần gọi hàm
    drawed_image = BLACK_IMAGE.copy()  

    # Tạo cửa sổ OpenCV và gán sự kiện chuột
    cv.namedWindow("Draw")
    cv.setMouseCallback("Draw", draw_number)

    while True:
        cv.imshow("Draw", drawed_image)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):  # Nhấn 'c' để xóa bảng vẽ
            drawed_image[:] = 0  
        elif key == ord('q'):  # Nhấn 'q' để thoát
            break

    # Thu nhỏ ảnh và đảm bảo đúng định dạng
    processed_image = np.squeeze(drawed_image)  # Bỏ chiều dư thừa (nếu có)
    processed_image = rescale(processed_image)
    
    return processed_image

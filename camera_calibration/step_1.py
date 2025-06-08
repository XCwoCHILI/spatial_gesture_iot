import cv2
import numpy as np
import os

# ==================== 1. 参数设置 ====================
chessboard_size = (8, 5)     # 棋盘格内角点数（列, 行）
square_size = 0.025          # 每格边长（单位：米）
capture_count = 0

# ==================== 2. 路径准备 ====================
save_folder = r"C:\Users\lenovo\Desktop\spatial_gesture_iot\camera_calibration"
image_folder = os.path.join(save_folder, "Captured_Images")
os.makedirs(image_folder, exist_ok=True)

# ==================== 3. 准备标定对象点和图像点 ====================
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints, imgpoints = [], []

# ==================== 4. 打开摄像头 ====================
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ 无法打开摄像头。")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📏 当前摄像头分辨率: {width} x {height}")
print("📸 请将棋盘格置于视野中，按 's' 保存，'q' 退出。")

# ==================== 5. 实时检测与保存 ====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取摄像头帧。")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 可视化检测结果
    if found:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, found)
        cv2.putText(frame, "Chessboard Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Chessboard NOT Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示图像（可设缩放比例）
    scale = 1.0
    display_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Camera Calibration (Press 's' to Save, 'q' to Quit)", display_frame)

    # 键盘操作：保存 or 退出
    key = cv2.waitKey(1)
    if key == ord('s') and found:
        capture_count += 1
        objpoints.append(objp)
        imgpoints.append(corners)

        image_path = os.path.join(image_folder, f"calibration_image_{capture_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"✅ 标定图像已保存: 第 {capture_count} 张 → {image_path}")

    elif key == ord('q'):
        break

# ==================== 6. 释放资源 ====================
cap.release()
cv2.destroyAllWindows()

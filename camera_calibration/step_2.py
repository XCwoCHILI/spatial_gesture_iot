import cv2
import numpy as np
import os
import cv2.aruco as aruco
import time

# ==================== 1. 设置保存路径和棋盘格参数 ==============
save_folder = r"C:\Users\lenovo\Desktop\spatial_gesture_iot\camera_calibration"
image_folder = os.path.join(save_folder, "Captured_Images")

chessboard_size = (8, 5)     # 内角点数量（列, 行）
square_size = 0.025          # 每个棋盘格的边长（米）
marker_length = 0.045        # ArUco 标记实际边长（米）

# ==================== 2. 读取保存的标定图像 ====================
if not os.path.exists(image_folder):
    print("❌ 未找到标定图像文件夹。请确保已保存标定图像。")
    exit()

image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
if not image_files:
    print("❌ 没有发现标定图像。")
    exit()

print(f"\n✅ 发现 {len(image_files)} 张标定图像，开始标定...")

# ==================== 3. 准备准备标定对象点和图像点 ============
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # 按实际大小缩放

objpoints, imgpoints = [], []

# ==================== 4. 遍历标定图像，自动检测角点 ============
for fname in image_files:
    img_path = os.path.join(image_folder, fname)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"✅ 检测到棋盘格角点：{fname}")
    else:
        print(f"❌ 未检测到棋盘格角点：{fname}")

#检查是否有有效标定数据
if len(objpoints) == 0:
    print("\n❌ 未检测到任何有效的标定数据，无法继续标定。")
    exit()

print(f"\n✅ 有效标定图像：{len(objpoints)} 张")

# ==================== 5. 执行相机标定 ====================
image_shape = gray.shape[::-1]
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_shape, None, None
)

# ==================== 6. 打印并保存标定结果 ====================
print("\n=== 标定结果 ===")
print("相机矩阵:\n", camera_matrix)
print("畸变系数:\n", dist_coeffs)
print("重投影误差:", ret)

np.save(os.path.join(save_folder, "camera_matrix.npy"), camera_matrix)
np.save(os.path.join(save_folder, "dist_coeffs.npy"), dist_coeffs)
np.save(os.path.join(save_folder, "objpoints.npy"), objpoints)
np.save(os.path.join(save_folder, "imgpoints.npy"), imgpoints)

print(f"✅ 标定完成，参数已保存至：{save_folder}")

# ==================== 7. 实时 ArUco 检测与位姿估计 ====================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters()
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.03)
            cv2.putText(frame, f"Id: {ids[i][0]}", (10, 40 + i * 60), font, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"rvec: {np.round(rvec[i].flatten(), 3)}", (10, 60 + i * 60), font, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"tvec: {np.round(tvec[i].flatten(), 3)}", (10, 80 + i * 60), font, 0.5, (255, 0, 0), 1)
    else:
        cv2.putText(frame, "No Ids", (10, 40), font, 1, (0, 255, 0), 2)

    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 120), font, 0.5, (0, 0, 255), 1)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:  # ESC 键退出
        print("退出程序...")
        break

cap.release()
cv2.destroyAllWindows()

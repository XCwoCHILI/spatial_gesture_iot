import cv2
import cv2.aruco as aruco
import mediapipe as mp
import numpy as np
import time
#这个程序是好的，可以跑的

# ============================= 相机参数 ============================= #
camera_matrix = np.load(r'C:\Users\lenovo\Desktop\spatial_gesture_iot\camera_calibration\camera_matrix.npy')
dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 假设无畸变

print("相机内参矩阵 (camera_matrix):\n", camera_matrix)
print("畸变系数 (dist_coeffs):\n", dist_coeffs)

# ============================= 初始化模块 ============================= #
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_params = aruco.DetectorParameters()
marker_length = 0.045  # ArUco 标签边长（单位：米）

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,            
                    smooth_landmarks=True,         
                    min_detection_confidence=0.5,  
                    min_tracking_confidence=0.5)   

# ============================= 打开摄像头 ============================= #
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX

# 初始化动作计时器
arm_extend_start_time = None
ACTIVATION_THRESHOLD = 3  # 秒

# 中指真实长度与相机焦距（用于深度估算）
L_REAL = 0.075
f = camera_matrix[0, 0]

# ============================= 主循环 ============================= #
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ===================== 设备检测（ArUco） ===================== #
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    # 初始化 P_dev_cam 默认值
    P_dev_cam = None

    if ids is not None:
        rvec_dev, tvec_dev, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        P_dev_cam = tvec_dev[0].reshape(3, 1)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_dev[0], tvec_dev[0], 0.03)
        cv2.putText(frame, "P_dev_cam: " + str(P_dev_cam.flatten()), (10, 50), font, 0.5, (0, 0, 255), 1)

    # ===================== 手部识别（MediaPipe） ===================== #
    results_hand = hands_detector.process(img_rgb)
    results_pose = pose_detector.process(img_rgb)

    # 初始化 P_wrist_cam 和 P_elbow_cam 默认值
    P_wrist_cam, P_elbow_cam = None, None
    # 初始化 z_finger 默认值
    z_finger = None

    if results_hand.multi_hand_landmarks and results_hand.multi_handedness:
        for idx, hand_landmarks in enumerate(results_hand.multi_hand_landmarks):
            label = results_hand.multi_handedness[idx].classification[0].label

            if label == 'Left':  # MediaPipe 将右手识别为 Left
        # 获取中指 mcp 和 tip 归一化坐标
                mcp = hand_landmarks.landmark[9]
                tip = hand_landmarks.landmark[12]
                
        # 获取中指 mcp 和 tip 图像像素坐标
                x1, y1 = int(mcp.x * w), int(mcp.y * h)
                x2, y2 = int(tip.x * w), int(tip.y * h)
                
        # 计算像素距离
                L_pixel = np.linalg.norm([x2 - x1, y2 - y1])

                # 根据像素长度估算中指所在深度
                if L_pixel > 0:
                    z_finger = (f * L_REAL) / L_pixel
                    cv2.putText(frame, f"Depth: {z_finger:.3f} m", (10, 70), font, 0.5, (0, 255, 0), 1)

                # 可视化中指连线和端点
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 5, (0, 0, 255), -1)

                # 获取手腕关键点（编号0）的归一化坐标
                wrist = hand_landmarks.landmark[0]
                u_wrist = int(wrist.x * w)
                v_wrist = int(wrist.y * h)
                
        # 计算右手腕的相机坐标系位置
                x = (u_wrist - camera_matrix[0, 2]) * z_finger / camera_matrix[0, 0]
                y = (v_wrist - camera_matrix[1, 2]) * z_finger / camera_matrix[1, 1]
                P_wrist_cam = np.array([[x], [y], [z_finger]])#假设右手腕深度与中指深度相同
                
                #显示右手腕的相机坐标
                cv2.putText(frame, "P_wrist_cam: " + str(P_wrist_cam.flatten()), (10, 90), font, 0.5, (0, 0, 255), 1)

                break

    # ===================== 姿态识别 ===================== #
    if results_pose.pose_landmarks and z_finger is not None:
        landmarks = results_pose.pose_landmarks.landmark
        world_landmarks = results_pose.pose_world_landmarks.landmark

        # 将关键点从图像坐标系转换为相机坐标系的转换函数
        def to_camera_coords(lm, z):
            u = lm.x * w
            v = lm.y * h
            x = (u - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
            y = (v - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
            return np.array([[x], [y], [z]])

        # 获得右手肘的坐标并连线至右手腕
        if landmarks[14].visibility > 0.25:
            
        # 使用 world 坐标估算右手肘的深度
            d_wrist = world_landmarks[16].z
            d_elbow = world_landmarks[14].z
            z_elbow = z_finger - (d_wrist - d_elbow)

        # 计算并显示右手肘的相机坐标系位置
            P_elbow_cam = to_camera_coords(landmarks[14], z_elbow)
            cv2.putText(frame, "P_elbow_cam: " + str(P_elbow_cam.flatten()), (10, 110), font, 0.5, (0, 0, 255), 1)

        # 获取像素坐标用于可视化连线 ？？为什么用像素坐标，应该直接用相机系下的坐标
            u_elbow = int(landmarks[14].x * w)
            v_elbow = int(landmarks[14].y * h)
            
        # 可视化右手肘与右手腕连线
            cv2.line(frame, (u_elbow, v_elbow), (u_wrist, v_wrist), (255, 255, 0), 2)
            cv2.circle(frame, (u_elbow, v_elbow), 5, (0, 255, 255), -1)
            cv2.circle(frame, (u_wrist, v_wrist), 5, (0, 255, 255), -1)

    # ===================== 空间交互逻辑 ===================== #
    if P_wrist_cam is not None and P_elbow_cam is not None and P_dev_cam is not None:
        
        # 计算手臂方向向量（从肘部指向手腕）
        arm_vec = P_wrist_cam - P_elbow_cam
        # 计算设备方向向量（从肘部指向设备）
        dev_vec = P_dev_cam - P_elbow_cam
        
        # 计算两向量夹角
        arm_unit = arm_vec / np.linalg.norm(arm_vec)
        dev_unit = dev_vec / np.linalg.norm(dev_vec)
        cos_theta = np.dot(arm_unit.T, dev_unit)[0][0]
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # 显示夹角信息
        cv2.putText(frame, f"Angle: {angle_deg:.1f}", (10, 30), font, 0.8, (255, 255, 0), 2)

        #如果夹角小于阈值，认为用户正在指向设备
        if angle_deg < 30:
            if arm_extend_start_time is None:
                # 首次检测到指向动作，记录起始时间
                arm_extend_start_time = time.time()
            else:
                # 计算指向动作持续时间
                elapsed = time.time() - arm_extend_start_time
                
                if elapsed < ACTIVATION_THRESHOLD:
                    # 指向时间未满足启动条件，显示“正在指向”状态
                    cv2.putText(frame, f"Pointing... ({elapsed:.1f}s)", (10, 400), font, 0.8, (0, 255, 255), 2)
                else:
                    # 满足持续时间，触发设备启动提示
                    device_id = int(ids[0][0])
                    cv2.putText(frame, f"device {device_id} start!", (10, 440), font, 1.0, (0, 0, 255), 3)
        else:
            # 角度超过阈值，重置动作计时器
            arm_extend_start_time = None
        

    # ===================== 显示帧率与图像 ===================== #
    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 460), font, 0.5, (0, 0, 255), 1)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:  # ESC 键退出
        print("ESC pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

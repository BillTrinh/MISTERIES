import cv2
import numpy as np

# 1. 配置文件路径（请确保文件完整且路径正确）
face_cascade_path = r'd:\computer-python code\computer_vision_project_beginner\haarcascade_frontalface_default.xml'
lbf_model_path = r'd:\computer-python code\computer_vision_project_beginner\lbfmodel.yaml'

# 2. 加载 Haar Cascade 人脸检测器
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print(f"错误：无法加载人脸检测模型，请检查路径：{face_cascade_path}")
    exit()

# 3. 初始化并加载 LBF 关键点检测器
facemark = cv2.face.createFacemarkLBF()
try:
    facemark.loadModel(lbf_model_path)
except cv2.error as e:
    print(f"错误：无法加载关键点模型。如果文件大小不对，请重新下载标准模型。路径：{lbf_model_path}")
    exit()

# 4. 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头。")
    exit()

print("程序已成功启动！按下 'q' 键退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧，程序退出。")
        break

    # 转换为灰度图以供人脸检测使用
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸（适当调整参数以提高准确度）
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=6, 
        minSize=(100, 100)
    )

    # 如果检测到人脸，开始预测关键点
    if len(faces) > 0:
        # 绘制人脸绿色矩形框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 预测人脸关键点
        success, landmarks = facemark.fit(gray, faces)

        if success:
            # 打印当前帧检测到的特征点数量（用于调试）
            print(f"检测到人脸数: {len(landmarks)}, 关键点数量: {len(landmarks[0][0])}")

            # 遍历每张脸的关键点数据
            for landmark in landmarks:
                # 将数据安全地重塑为 (N, 2) 的二维坐标阵列
                pts = np.array(landmark, dtype=np.int32).reshape(-1, 2)
                
                # 遍历所有点并进行绘制
                for idx, (x_p, y_p) in enumerate(pts):
                    # 画出蓝色实心圆点
                    cv2.circle(frame, (int(x_p), int(y_p)), 2, (255, 0, 0), -1)
                    
                    # 在点旁边绘制黄色的序号数字 (0-67)
                    cv2.putText(
                        frame, 
                        str(idx), 
                        (int(x_p) + 2, int(y_p) + 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.3, 
                        (0, 255, 255), 
                        1
                    )

    # 展示实时画面
    cv2.imshow('Facial Keypoints Detection', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并销毁所有窗口
cap.release()
cv2.destroyAllWindows()
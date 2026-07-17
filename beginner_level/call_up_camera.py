import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    window_name = 'Face Keypoints'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    #  通过这个布尔变量控制是否左右翻转
    # True 1  = 开启左右翻转（镜面效果）
    # False 0 = 关闭左右翻转（正常相机视角）
    mirror_mode = 0 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 根据布尔变量决定是否执行翻转
        if mirror_mode:
            frame = cv2.flip(frame, 1)
            
        cv2.imshow(window_name, frame)

        # 判断窗口是否被点击了 X 关闭
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0:
            break

        # 按 'q' 键退出
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
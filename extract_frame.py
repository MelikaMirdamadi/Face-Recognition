import cv2

cap = cv2.VideoCapture("rtsp://localhost:8554/cctv1")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # اینجا می‌توان روی هر فریم تشخیص چهره انجام داد
    cv2.imshow("RTSP Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2

username = "admin"
password = "tnwcss12"
ip = "192.168.1.67"
rtsp_url = f"rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=0"


def preview_rtsp_stream() -> None:
    """Open the RTSP stream in a simple OpenCV preview window."""


    print("Attempting to connect...")
    params = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
    cap = cv2.VideoCapture()
    cap.open(rtsp_url, cv2.CAP_FFMPEG, params)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("RTSP connection failed.")
        return

    print("Connected successfully!")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame drop occurred.")
                break
            cv2.imshow("Dahua Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    preview_rtsp_stream()

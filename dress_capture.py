import cv2

def capture_image():
    cap = cv2.VideoCapture(0)  
    while True:
        ret, frame = cap.read()
        cv2.imshow("Press 's' to capture, 'q' to quit", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):  
            cv2.imwrite("captured_dress.jpg", frame)
            print("Image Captured: captured_dress.jpg")
            break
        elif key == ord('q'):  
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_image()

import cv2 as cv
import os
import threading

# Create a directory for reference images
output_dir = 'ReferenceImages'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def capture_reference_image(object_name):
    cap = cv.VideoCapture(0)
    
    # Set camera properties to improve performance
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Width
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Height

    print(f"Press 'c' to capture an image of the {object_name}. Press 'q' to quit.")

    def display_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image.")
                break

            cv.imshow('Capture Reference Image', frame)

            key = cv.waitKey(1)
            if key & 0xFF == ord('c'):
                image_path = os.path.join(output_dir, f'{object_name}_reference.png')
                cv.imwrite(image_path, frame)
                print(f"Image captured: {image_path}")
            elif key & 0xFF == ord('q'):
                break

    # Start the frame display in a separate thread
    thread = threading.Thread(target=display_frames)
    thread.start()

    thread.join()  # Wait for the thread to finish

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    object_name = input("Enter the object name (e.g., 'person', 'cell phone'): ")
    capture_reference_image(object_name)

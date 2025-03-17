import cv2 as cv
import numpy as np
import pyttsx3
from ultralytics import YOLO

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Load YOLOv8 Nano model
class_names = model.names  # Get class names

# Define object dimensions in cm
PERSON_WIDTH_CM = 40.64  # 16 inches converted to cm (16 * 2.54)
MOBILE_WIDTH_CM = 7.62   # 3 inches converted to cm (3 * 2.54)

# Initialize video capture
cap = cv.VideoCapture(0)

# Focal length variables (assumed to be calculated beforehand)
focal_person = 800  # Placeholder value; replace with actual calculated value
focal_mobile = 400  # Placeholder value; replace with actual calculated value

# Open file to save detection results
output_file = open("output.txt", "w")

# Main loop for video capture
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the current frame
    results = model(frame)
    announcements = []  # List to store announcements

    for result in results:
        for detection in result.boxes:
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])  # Confidence score
            x1, y1, x2, y2 = map(int, bbox)  # Bounding box coordinates
            object_width_in_frame = x2 - x1
            object_name = class_names[class_id]

            # Draw bounding box and label
            color = (0, 255, 0)
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{object_name}: {confidence:.2f}"
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

            # Distance estimation logic (in cm)
            distance_cm = None
            if object_name == 'person':
                distance_cm = (PERSON_WIDTH_CM * focal_person) / object_width_in_frame
            elif object_name == 'cell phone':
                distance_cm = (MOBILE_WIDTH_CM * focal_mobile) / object_width_in_frame

            if distance_cm is not None and object_width_in_frame > 0:  # Prevent division by zero
                # Draw the distance on the frame
                cv.putText(frame, f"Distance: {round(distance_cm, 2)} cm", 
                           (x1 + 5, y1 + 20), cv.FONT_HERSHEY_COMPLEX, 0.48, (0, 255, 0), 2)

                # Announce detected object and distance
                announcement = f"{object_name} detected at {round(distance_cm, 2)} centimeters"
                announcements.append(announcement)

                # Save detection details to file
                output_file.write(f"{object_name} detected at {round(distance_cm, 2)} cm with {confidence:.2f} confidence\n")

    # Process the speech queue after processing all detections
    if announcements:  # Only process if there are announcements
        engine = pyttsx3.init()  # Reinitialize TTS engine to prevent errors
        for announcement in announcements:
            engine.say(announcement)
        engine.runAndWait()
        engine.stop()

    # Show the video frame
    cv.imshow('YOLOv8 Distance Estimation Demo', frame)

    # Quit with 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break  # ✅ 'break' is now correctly inside the while loop

# Cleanup
cap.release()
cv.destroyAllWindows()
output_file.close()  # Close the file after finishing
print("Detection results saved to output.txt")

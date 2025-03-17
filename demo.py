import cv2 as cv
import numpy as np
import pyttsx3
import webcolors
from ultralytics import YOLO

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
class_names = model.names

# Define object dimensions in cm
PERSON_WIDTH_CM = 40.64  # 16 inches in cm
MOBILE_WIDTH_CM = 7.62   # 3 inches in cm

# Focal length values (adjust as needed)
focal_person = 800  
focal_mobile = 400  

# CSS3 color mapping
CSS3_NAMES_TO_HEX = {
    "black": "#000000", "silver": "#C0C0C0", "gray": "#808080", "white": "#FFFFFF",
    "maroon": "#800000", "red": "#FF0000", "purple": "#800080", "fuchsia": "#FF00FF",
    "green": "#008000", "lime": "#00FF00", "olive": "#808000", "yellow": "#FFFF00",
    "navy": "#000080", "blue": "#0000FF", "teal": "#008080", "aqua": "#00FFFF"
}

# Function to estimate distance
def estimate_distance(object_name, object_width_in_frame):
    if object_name == 'person':
        return (PERSON_WIDTH_CM * focal_person) / object_width_in_frame
    elif object_name == 'cell phone':
        return (MOBILE_WIDTH_CM * focal_mobile) / object_width_in_frame
    return None  # Unknown object

# Function to find the closest CSS color name
def closest_css_color(requested_color):
    min_colors = {}
    for name, hex_code in CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# Function to get the dominant color of an object
# def get_dominant_color(image, bbox):
#     x1, y1, x2, y2 = map(int, bbox)
#     object_img = image[y1:y2, x1:x2]
#     average_color = np.mean(object_img, axis=(0, 1)).astype(int)
#     return closest_css_color(average_color)
from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
import webcolors

def get_dominant_color(image, bbox, object_name):
    x1, y1, x2, y2 = map(int, bbox)

    if object_name == "person":
        # Focus on the central-lower part of the bounding box (avoiding the face)
        y1 = y1 + int((y2 - y1) * 0.4)  # Skip upper 40% (head & shoulders)
    
    object_img = image[y1:y2, x1:x2]

    if object_img.size == 0:
        return "unknown"

    # Convert to LAB color space for better color perception
    object_img = cv.cvtColor(object_img, cv.COLOR_BGR2LAB)

    # Reshape the image for clustering
    pixels = object_img.reshape((-1, 3))

    # Use K-Means clustering to find the most dominant color
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)  # Use 3 clusters
    kmeans.fit(pixels)

    # Find the largest cluster
    cluster_sizes = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(cluster_sizes)].astype(int)

    # Convert back to RGB for color matching
    dominant_color_rgb = cv.cvtColor(np.uint8([[dominant_color]]), cv.COLOR_LAB2RGB)[0][0]

    return closest_css_color(dominant_color_rgb)


# Initialize video capture
cap = cv.VideoCapture(0)

# Open file to save detection results
output_file = open("output.txt", "w")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    results = model(frame)
    announcements = []

    for result in results:
        for detection in result.boxes:
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            x1, y1, x2, y2 = map(int, bbox)
            object_width_in_frame = x2 - x1
            object_name = class_names[class_id]

            # Get object color
            object_color = get_dominant_color(frame, bbox, object_name)


            # Estimate distance
            distance_cm = estimate_distance(object_name, object_width_in_frame)

            # Draw bounding box and label
            color = (0, 255, 0)
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{object_name} ({object_color}): {confidence:.2f}"
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

            if distance_cm:
                distance_text = f"Distance: {round(distance_cm, 2)} cm"
                cv.putText(frame, distance_text, (x1 + 5, y1 + 20), cv.FONT_HERSHEY_COMPLEX, 0.48, (0, 255, 0), 2)

                # Prepare text-to-speech output
                announcement = f"{object_name} ({object_color}) detected at {round(distance_cm, 2)} centimeters"
                announcements.append(announcement)

                # Save detection details to file
                output_file.write(f"{object_name} ({object_color}) detected at {round(distance_cm, 2)} cm with {confidence:.2f} confidence\n")

    # Process the speech queue
    if announcements:
        engine = pyttsx3.init()
        for announcement in announcements:
            engine.say(announcement)
        engine.runAndWait()
        engine.stop()

    # Display the video feed
    cv.imshow('Object Detection with Distance & Color', frame)

    # Exit condition
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
output_file.close()
print("Detection results saved to output.txt")

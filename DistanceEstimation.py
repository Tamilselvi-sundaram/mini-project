import pandas as pd
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import webcolors

# Manual CSS3 color hex mapping
CSS3_NAMES_TO_HEX = {
    "black": "#000000", "silver": "#C0C0C0", "gray": "#808080", "white": "#FFFFFF",
    "maroon": "#800000", "red": "#FF0000", "purple": "#800080", "fuchsia": "#FF00FF",
    "green": "#008000", "lime": "#00FF00", "olive": "#808000", "yellow": "#FFFF00",
    "navy": "#000080", "blue": "#0000FF", "teal": "#008080", "aqua": "#00FFFF",
    # Add more if needed
}

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the appropriate YOLOv8 model

# Function to estimate distance
def estimate_distance(bbox):
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return 1000 / bbox_area if bbox_area != 0 else 0

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

# Function to get the dominant color
def get_dominant_color(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    object_img = image[y1:y2, x1:x2]
    average_color = np.mean(object_img, axis=(0, 1)).astype(int)
    css_color_name = closest_css_color(average_color)
    return css_color_name

# Initialize results storage
results_list = []

# Open video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    for result in results:
        for detection in result.boxes:
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            class_name = model.names[class_id]

            # Estimate distance
            distance_cm = estimate_distance(bbox)

            # Get dominant color
            dominant_color = get_dominant_color(frame, bbox)

            # Store the result in the list
            results_list.append({
                'Object': class_name,
                'Distance (cm)': distance_cm,
                'Dominant Color': dominant_color
            })

            # Display bounding box, class, and distance on the image
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {distance_cm:.2f} cm, Color: {dominant_color}",
                        (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

output_path = r'C:\Users\91887\Downloads\object_detection_results.xlsx'  # Ensure 'C:\temp' exists or choose another writable directory
df.to_excel(output_path, index=False)

print(results_list)  # Add this to see if the list contains data

for result in results:
    for detection in result.boxes:
        bbox = detection.xyxy[0].cpu().numpy()
        class_id = int(detection.cls[0])
        class_name = model.names[class_id]
        
        print(f"Detected: {class_name}, BBox: {bbox}")  # Debug print to verify detection

print(f"Results list contains {len(results_list)} entries")

sample_data = [{'Object': 'Test', 'Distance (cm)': 50, 'Dominant Color': 'red'}]
df = pd.DataFrame(sample_data)
df.to_excel(r'C:\Users\91887\Downloads\object_detection_resultS.xlsx', index=False)

print("Sample data saved to Excel")

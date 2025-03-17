import cv2
import numpy as np

def get_dominant_color(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    roi = roi.reshape((-1, 3))
    pixels, counts = np.unique(roi, axis=0, return_counts=True)
    dominant_color = pixels[np.argmax(counts)]
    
    return tuple(dominant_color)

def identify_color_name(rgb_color):
    colors = {
        "Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0), "Cyan": (0, 255, 255), "Magenta": (255, 0, 255),
        "Black": (0, 0, 0), "White": (255, 255, 255), "Gray": (128, 128, 128)
    }
    min_distance = float("inf")
    closest_color = "Unknown"
    
    for name, color in colors.items():
        distance = np.linalg.norm(np.array(color) - np.array(rgb_color))
        if distance < min_distance:
            min_distance = distance
            closest_color = name
    
    return closest_color

# Example usage
image = cv2.imread("image.jpg")  # Load your image
bbox = (50, 50, 100, 100)  # Example bounding box (x, y, width, height)
color = get_dominant_color(image, bbox)
color_name = identify_color_name(color)
print("Detected Color:", color_name)
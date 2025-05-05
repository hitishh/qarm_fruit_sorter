# Fruit Centerline Detection and Gripper Line Drawing with Angle, Direction, and Auto Saving
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import os
import math

# Global to store fruit name for saving
fruit_name = "fruit"

def detect_and_crop_fruit(image, fruit_type):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if fruit_type == 'banana':
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    elif fruit_type == 'tomato':
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([179, 255, 255])
    elif fruit_type == 'strawberry':
        lower = np.array([0, 100, 100])
        upper = np.array([10, 255, 255])
    else:
        raise ValueError("Unknown fruit type!")

    if fruit_type == 'tomato':
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No fruit detected, using original image.")
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped = image[y:y+h, x:x+w]
    return cropped

def calculate_alignment_angle_and_draw(image, show_plot=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No object detected!")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.reshape(-1, 2).astype(np.float32)

    if len(points) < 5:
        print("Not enough points for PCA. Skipping.")
        return None

    mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    principal_direction = eigenvectors[0]
    gripper_direction = np.array([-principal_direction[1], principal_direction[0]])

    com = center_of_mass(cleaned)
    com_y, com_x = int(com[0]), int(com[1])

    # Calculate angle and direction
    dx, dy = gripper_direction
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 180
    direction = "clockwise" if 0 <= angle_deg <= 90 else "anticlockwise"
    print(f"Angle: {angle_deg:.1f}°")
    print(f"Direction: {direction}")

    if show_plot:
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        height, width = output.shape[:2]
        line_length = max(height, width) * 2

        x1 = int(com_x - line_length * gripper_direction[0])
        y1 = int(com_y - line_length * gripper_direction[1])
        x2 = int(com_x + line_length * gripper_direction[0])
        y2 = int(com_y + line_length * gripper_direction[1])

        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.drawMarker(output, (com_x, com_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

        # Draw angle and direction text (black, smaller, 2 lines)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_x = 10
        text_y = 30
        cv2.putText(output, f"Angle: {angle_deg:.1f}°", (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(output, f"Direction: {direction}", (text_x, text_y + 30), font, font_scale, (0, 0, 0), thickness)

        plt.figure(figsize=(8, 10))
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title("Gripper Line and Orientation")
        plt.axis('off')
        plt.show()

        # Save the image to specified path
        save_dir = r"C:\Users\hitis\Desktop\Applied Robotics\Report\Codes\Fully Auto\Center_line\Results"
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{fruit_name}_{angle_deg:.1f}deg_{direction}.jpg"
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, output)
        print(f"Saved output image to: {save_path}")

    return gripper_direction

def process_fruit_image(image_path, fruit_type):
    global fruit_name
    fruit_name = fruit_type  # Store globally for filename use

    if not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist!")
        return

    image = cv2.imread(image_path)
    cropped = detect_and_crop_fruit(image, fruit_type)
    gripper_direction = calculate_alignment_angle_and_draw(cropped)
    if gripper_direction is not None:
        print(f"Gripper direction vector: {gripper_direction}")

if __name__ == "__main__":
    while True:
        input_image_path = input("Enter the path to the image (or 'q' to quit): ").strip()
        if input_image_path.lower() == 'q':
            print("Exiting program.")
            break

        fruit = input("Enter the fruit type (banana, tomato, strawberry): ").strip().lower()
        process_fruit_image(input_image_path, fruit)

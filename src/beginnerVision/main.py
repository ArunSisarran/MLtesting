"""
NYC Street Object Detection using YOLOv8
----------------------------------------
This script demonstrates how to use YOLOv8 for object detection in NYC street scenes.
It includes functions for:
1. Processing a single image
2. Real-time webcam detection
3. Analyzing and visualizing results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os

# Install required packages if they're not already installed
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "ultralytics"])
    subprocess.check_call(["pip", "install", "opencv-python"])
    subprocess.check_call(["pip", "install", "numpy", "matplotlib"])
    from ultralytics import YOLO


def download_model():
    """Download the pre-trained YOLOv8 model"""
    print("Loading pre-trained YOLOv8 model (will download if not present)...")
    model = YOLO('yolov8n.pt')  # nano model (smallest and fastest)
    print("Model loaded successfully!")
    return model


def download_sample_image(url=None):
    """Download a sample NYC street image if none is provided"""
    if url is None:
        # Default sample NYC street image - using a more reliable source
        url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
        # Note: This is not actually an NYC street image, but it's a reliable test image
    try:
        print(f"Downloading sample image from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses

        img = Image.open(BytesIO(response.content))

        # Save the image
        img_path = "sample_image.jpg"
        img.save(img_path)
        print(f"Sample image saved to {img_path}")

        return img_path

    except Exception as e:
        print(f"Error downloading image: {e}")
        print("Using a local sample image instead...")

        # Create a simple test image if download fails
        img = Image.new('RGB', (640, 480), color=(73, 109, 137))
        img_path = "sample_image.jpg"
        img.save(img_path)

        return img_path


def detect_objects_in_image(model, image_path, confidence_threshold=0.55):
    """
    Perform object detection on a single image
    Args:
        model: The loaded YOLO model
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score for detections (0-1)

    Returns:
        None (displays and saves results)
    """
    print(f"Performing object detection on {image_path}...")

    # Run inference on the image with the specified confidence threshold
    results = model(image_path, conf=confidence_threshold)
    # Get the first result (assuming only one image was passed)
    result = results[0]
    # Visualize the results (result.plot() returns an annotated image as a numpy array)
    annotated_img = result.plot()
    # Display the annotated image
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Object Detection Results - YOLOv8")

    # Save the annotated image
    output_path = "detected_" + os.path.basename(image_path)
    cv2.imwrite(output_path, annotated_img)
    print(f"Annotated image saved to {output_path}")

    # Show the plot
    plt.show()

    # Print the detected objects and their confidence scores
    print("\nDetected objects:")
    object_counts = {}

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        # Count objects by class
        if class_name in object_counts:
            object_counts[class_name] += 1
        else:
            object_counts[class_name] = 1

        # Print each detection
        print(f"- {class_name}: {confidence:.2f}")

    # Print summary
    print("\nSummary:")
    for obj_class, count in object_counts.items():
        print(f"- {obj_class}: {count}")

    return result


def detect_from_webcam(model, confidence_threshold=0.25):
    """
    Perform real-time object detection using webcam feed
    Args:
        model: The loaded YOLO model
        confidence_threshold: Minimum confidence score for detections (0-1)
    Returns:
        None
    """
    print("Starting webcam detection. Press 'q' to quit...")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Perform object detection
        results = model(frame, conf=confidence_threshold)

        # Visualize the results
        annotated_frame = results[0].plot()

        # Add instructions text
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")


def main():
    """Main function to run the object detection demo"""
    print("NYC Street Object Detection using YOLOv8")
    print("---------------------------------------")
    # Load the YOLO model
    model = download_model()

    # Menu for user to choose detection mode
    while True:
        print("\nChoose an option:")
        print("1. Detect objects in a sample NYC street image")
        print("2. Detect objects in your own image")
        print("3. Real-time detection using webcam")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            # Download and analyze a sample image
            sample_img_path = download_sample_image()
            detect_objects_in_image(model, sample_img_path)

        elif choice == '2':
            # Ask user for their own image path
            user_img_path = input("Enter the path to your image: ")
            if os.path.exists(user_img_path):
                detect_objects_in_image(model, user_img_path)
            else:
                print(f"Error: Image not found at {user_img_path}")

        elif choice == '3':
            # Start webcam detection
            detect_from_webcam(model)

        elif choice == '4':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from ultralytics import NAS

# Ensure you have the correct path to the pretrained model
model_path = "yolo_nas_s.pt"

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Please ensure the file is downloaded and the path is correct.")

# Instantiate the YOLO-NAS model
model = NAS(model_path)

# Process single image for skeleton display
def process_single_image(image, results):
    annotated_image = results[0].plot()
    return annotated_image

# Main function to run the model on a local video file
def main(input_video_path):
    print("Starting video processing...")

    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('skeleton_output.mp4', fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run prediction on the current frame
        results = model(frame)

        # Process the prediction
        skeleton_frame = process_single_image(frame, results)

        # Write the frame with skeleton overlay to output video
        out.write(skeleton_frame)

        # Display the frame in real-time
        cv2.imshow('Skeleton Overlay', skeleton_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed. Output saved as 'skeleton_output.mp4'.")

if __name__ == "__main__":
    input_video_path = "D:/multicam/RDJ.mp4"
    main(input_video_path)

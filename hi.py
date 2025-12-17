import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Load YOLO TFLite model
interpreter = tf.lite.Interpreter(model_path ="E:\Vs_code_python\yolo.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to run object detection
def run_inference(image):
    # Preprocess image
    image_resized = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    image_resized = np.expand_dims(image_resized, axis=0)
    image_resized = np.float32(image_resized)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_resized)
    interpreter.invoke()

    # Get output tensor
    boxes = interpreter.get_tensor(output_details[0]['index'])
    class_ids = interpreter.get_tensor(output_details[1]['index'])
    confidences = interpreter.get_tensor(output_details[2]['index'])

    return boxes, class_ids, confidences

# Display Streamlit UI
st.title("YOLO Object Detection with TFLite")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Run inference on the uploaded image
    boxes, class_ids, confidences = run_inference(image)

    # Display results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Detected {len(boxes)} objects")

    # Display bounding boxes on the image
    for box, class_id, confidence in zip(boxes[0], class_ids[0], confidences[0]):
        if confidence > 0.5:  # Filter out low-confidence detections
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * image.shape[1]), int(y1 * image.shape[0]), int(x2 * image.shape[1]), int(y2 * image.shape[0])

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            st.text(f"Class ID: {class_id} | Confidence: {confidence}")

    # Display the image with bounding boxes
    st.image(image, caption="Image with Detections", use_column_width=True)

# Video upload (optional)
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Read video
    video_bytes = video_file.read()
    video_stream = io.BytesIO(video_bytes)
    cap = cv2.VideoCapture(video_stream)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on each frame
        boxes, class_ids, confidences = run_inference(frame)

        # Display bounding boxes on the frame
        for box, class_id, confidence in zip(boxes[0], class_ids[0], confidences[0]):
            if confidence > 0.5:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

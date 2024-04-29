import cv2
import concurrent.futures
from roboflow import Roboflow
from tqdm import tqdm
import streamlit as st
import tempfile
import os
temp_path = 'temp.mp4'
# Initialize Roboflow and get the model
rf = Roboflow(api_key="lhptDCEBHRYiaI88pjs2")
project = rf.workspace().project("trajectory-detection")
model = project.version(1).model
# Function to process each frame
def process_frame(image):
    p = model.predict(image, confidence=40, overlap=30).json()
    if p['predictions']:
        x = int(p['predictions'][0]['x'])
        y = int(p['predictions'][0]['y'])
        return x, y
    return None
# Define the Streamlit app code
@st.cache_data
def process_video(video_path):
    # Read the video file
    vidcap = cv2.VideoCapture(video_path)
    # Get the frame rate of the original video
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    # Store the modified frames in a list
    modified_frames = []
    # Iterate over each frame in the video
    success, image = vidcap.read()
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")
    while success:
        modified_frames.append(image)
        success, image = vidcap.read()
        pbar.update(1)
    # Release the VideoCapture object
    vidcap.release()
    # Process the frames in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_frame, modified_frames)
    # Get the frame size from the first modified frame
    frame_height, frame_width, _ = modified_frames[0].shape
    # Create a VideoWriter object with the original frame rate
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4").name
    output_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
    # Store the previous 5 predictions
    prev_predictions = [None] * 5
    # Iterate over the results and modified frames
    for i, result in enumerate(results):
        if result is not None:
            x, y = result
            dot_color = (0, 255, 0)  # BGR color tuple (green dot in this example)
            dot_radius = 10  # Radius of the dot in pixels
            image = modified_frames[i]
            # Update the sliding window of previous predictions
            prev_predictions.append((x, y))
            prev_predictions.pop(0)
            # Mark the predictions of the previous 5 frames
            for j in range(5):
                if j >= 0 and prev_predictions[j] is not None:
                    prev_x, prev_y = prev_predictions[j]
                    cv2.circle(image, (prev_x, prev_y), dot_radius, dot_color, -1)
            # Mark the current prediction
            cv2.circle(image, (x, y), dot_radius, dot_color, -1)
        # Write the modified frame to the output video
        output_video.write(image)
        pbar.set_postfix({"Done": i + 1, "Remaining": total_frames - (i + 1)})
        pbar.update(1)
    # Release the VideoWriter object
    output_video.release()
    pbar.close()
    # Return the output video file path
    return output_file
# Set Streamlit app title
st.title("Trajectory Detection Dashboard")
# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
# Process video frames and display download button
if uploaded_file is not None:
    video_path = temp_path  # Save the uploaded file temporarily
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.write("Processing video... (This may take a while)")
    output_file = process_video(video_path)
    st.write("Video processing complete!")
    # Create a download button for the processed video
    st.download_button(
        label="Download Processed Video",
        data=open(output_file, 'rb').read(),
        file_name="processed_video.mp4",
        mime="video/mp4"
    )
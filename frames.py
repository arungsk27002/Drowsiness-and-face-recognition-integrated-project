import cv2
import os

# Open the video file
video_path = 'VID-20240309-WA0012.mp4'
cap = cv2.VideoCapture(video_path)

# Create a folder to save the frames
output_folder = 'aswin'
os.makedirs(output_folder, exist_ok=True)

# Extract and save each frame
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame_filename = f'frame_{frame_count}.jpg'
    frame_path = os.path.join(output_folder, frame_filename)
    cv2.imwrite(frame_path, frame)

# Release the video capture object
cap.release()

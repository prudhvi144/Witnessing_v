
"""
Script Name: Read AVI Video Dimensions
Description: This script reads all the AVI video files of any size from Embryoscope in a directory and prints their dimensions to the console.
Author: Prudhvi
Date: March 8, 2023
Version: 1.0
"""
import cv2
import os

from tqdm import tqdm

input_folder = '../../data/processed2/'
output_folder = '../../data/mouse_frames/'

os.makedirs(output_folder, exist_ok=True)



video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

for video_file in tqdm(video_files):

    video_name = os.path.splitext(video_file)[0]
    # print (video_name)
    # Create a new folder with the same name as the video file
    output_folder_path = os.path.join(output_folder, video_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    video_path = os.path.join(input_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file {video_file}")
        continue
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):

        ret, frame = cap.read()


        if not ret:
            print(f"Could not read frame {i} from video file {video_file}")
            break

        frame_path = os.path.join(output_folder_path, f"{i}.jpg")
        cv2.imwrite(frame_path, frame)


    frame = cv2.imread(os.path.join(output_folder_path, "0.jpg"))
    mean_color = int(cv2.mean(frame)[0])

    # Print the frame number and its color
    # print(f"Frame {i}: Mean color = {mean_color}")
    # if frame is not None and mean_color == 250:
    #     # If the entire video is just white, delete the video and its corresponding folder
    #     os.remove(video_path)
    #     os.removedirs(output_folder_path)

    # Release the video capture object
    cap.release()

"""
Script Name: Read AVI Video Dimensions
Description: This script reads all the AVI video files in a directory and prints their dimensions to the console.
Author: Prudhvi
Date: March 8, 2023
Version: 1.0
"""

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ImageClip
import threading

import os

folder_path = '../../data/raw/12/'
output_folder = '../../data/processed/'

video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi') or f.endswith('.mp4')]


for video_file in video_files:
    try:

        cap = cv2.VideoCapture(os.path.join(folder_path, video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width, out_height = 250, 250
        out_fps = fps


        file_name = os.path.splitext(video_file)[0]
        print(f"file_name: {file_name}")
        out_files = [cv2.VideoWriter(os.path.join(output_folder, f'{file_name}_{i}.mp4'), fourcc, out_fps, (out_width, out_height)) for i in [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]


        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the frame into 12 equal parts
            for i in range(3):
                for j in range(4):
                    x = j * out_width
                    y = height - ((i + 1) * out_height)
                    cropped_frame = frame[y:y+out_height, x:x+out_width]
                    # Write the cropped frame to the corresponding output video
                    out_file_index = i * 4 + j
                    out_file_name = f'{file_name}_{out_file_index}.mp4'
                    out_file = out_files[out_file_index]
                    out_file.write(cropped_frame)

        cap.release()
        for out_file in out_files:
            out_file.release()
    except Exception as e:
        print(f"Ignoring {video_file} due to error: {e}")
        continue


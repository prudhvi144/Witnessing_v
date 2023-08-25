"""
Script Name: filter
Description: This script used to delete empty wells.
Author: Prudhvi
Date: March 8, 2023
Version: 1.0
"""
import cv2
import os

folder_path = '../../data/processed/'

for filename in os.listdir(folder_path):
    if filename.endswith('.mp4') or filename.endswith('.avi'):
        filepath = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        cap.release()
        mean_color = cv2.mean(frame)[0]
        if mean_color == 250:
            os.remove(filepath)
            print(f"{filename} deleted")
        else:
            print(f"{filename} not deleted")
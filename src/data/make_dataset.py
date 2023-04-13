import os
import random
import csv

input_dir = '../../data/interim.v1/'
output_dir = '../../data/external/'


def write_file_list(file_path, file_list):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'class', 'imagescore'])
        for subdir in file_list:
            subdir_path = os.path.join(input_dir, subdir)
            for root, dirs, files in os.walk(subdir_path):
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                        class_name = subdir
                        writer.writerow([os.path.join(root.replace("../data","data"), file), class_name, os.path.splitext(os.path.basename(file))[0]])

folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

print(len(folders))

random.shuffle(folders)

train_size = int(len(folders) * 0.7)
val_size = int(len(folders) * 0.2)
test_size = len(folders) - train_size - val_size
train_folders = folders[:train_size]
val_folders = folders[train_size:train_size+val_size]
test_folders = folders[train_size+val_size:]

write_file_list(os.path.join(output_dir, 'train.csv'), train_folders)
write_file_list(os.path.join(output_dir, 'val.csv'), val_folders)
write_file_list(os.path.join(output_dir, 'test.csv'), test_folders)

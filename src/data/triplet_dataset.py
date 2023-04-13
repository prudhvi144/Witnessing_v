import os

from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
import torch
import csv


def get_transforms(resize_size=256, crop_size=224):
    return Compose([
        Resize(resize_size),
        # RandomCrop(crop_size),
        # RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def rgb_loader(path):
    # path = path[3:]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class TripletDataset(Dataset):

    def getdata(self, filename):
        data = {}  # dictionary to store data with headers as keys
        # filename = filename[3:]
        with open(filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)
            for row in csv_reader:
                for i in range(len(row)):
                    key = header[i]
                    if key not in data:
                        data[key] = []
                    data[key].append(row[i])
        return data

    def __init__(self, params, mode, transform=None, score_diff=10):

        print(os.listdir("./"))

        self.data = self.getdata(os.path.join("../", "data",  params.dataset_name, mode+'.v1.csv'))
        self.mode = mode
        self.transform = get_transforms()
        self.loader = rgb_loader
        self.score_diff = score_diff

        if self.mode == 'train':
            self.train_labels = self.data['class']
            self.train_path = self.data['path']
            self.labels_set = set(np.array(self.train_labels))
            self.label_to_indices = {label: np.where(np.array(self.train_labels)== label)[0]
                                     for label in self.labels_set}
            self.scores = self.data['imagescore']


        elif self.mode == 'test' or self.mode == 'val':

            self.test_labels = self.data['class']
            self.test_data = self.data["path"]
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                     for label in self.labels_set}
            self.scores = self.data['imagescore']
            self.scores = [int(i) for i in self.scores]
            self.triplets = []
            self.triplet_labels = []
            n = len(self.test_data)
            arr = np.arange(n)
            ## arr shape = (n, )
            for i in range(0, n, 3):
                try:
                    self.triplets.append([arr[i], arr[i+1], arr[i+2]])
                    self.triplet_labels.append(self.test_labels[i])
                except IndexError:
                    break




            # triplets = np.reshape(arr, (3,))

            # check if n is divisible by 3
            # if n % 3 != 0:
            #     # delete a sample to make it divisible
            #     triplets = np.delete(self.triplets, -1, axis=0)
            #
            # # reshape the array to -1,3
            # self.triplets = np.reshape(triplets, (-1, 3))
            # print((self.triplets))



        elif self.mode == 'validation':
            self.val_labels = self.data['class']
            self.val_data = self.data['path']
            self.labels_set = set(np.array(self.val_labels))
            self.label_to_indices = {label: np.where(np.array(self.val_labels)== label)[0]
                                        for label in self.labels_set}

            self.scores = self.data['imagescore']
            self.scores = [int(i) for i in self.scores]

            triplets = []
            for i in range(len(self.val_data)):
                anchor_index = i
                anchor_label = self.labels[anchor_index]
                positive_label = anchor_label
                while positive_label == anchor_label:
                    positive_index = np.random.choice(self.label_to_indices[positive_label])
                    if abs(self.scores[anchor_index] - self.scores[positive_index]) < self.score_diff:
                        break

                negative_label = np.random.choice(list(self.labels_set - {anchor_label}))
                negative_index = np.random.choice(self.label_to_indices[negative_label])

                triplets.append([anchor_index, positive_index, negative_index])

            self.triplets = triplets


    def __getitem__(self, index):
        if self.mode == 'train':
            anchor_index = index
            anchor_label = (self.train_labels[anchor_index])
            positive_label = anchor_label
            while positive_label == anchor_label:
                positive_index = np.random.choice(self.label_to_indices[positive_label])
                if abs(int(self.scores[anchor_index]) - int(self.scores[positive_index])) <= int(self.score_diff):
                    break
            else:
                positive_index = np.random.choice(self.label_to_indices[positive_label])

            negative_label = np.random.choice(list(self.labels_set - {anchor_label}))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            anchor_img = self.loader(self.train_path[anchor_index])
            positive_img = self.loader(self.train_path[positive_index])
            negative_img = self.loader(self.train_path[negative_index])
            self.triplet_label_index = [anchor_label, positive_label, negative_label]
        else:
            anchor_img = self.loader(self.test_data[self.triplets[index][0]])
            positive_img = self.loader(self.test_data[self.triplets[index][1]])
            negative_img = self.loader(self.test_data[self.triplets[index][2]])
            path1 = (self.test_data[self.triplets[index][0]])
            path2 = (self.test_data[self.triplets[index][1]])
            path3 = (self.test_data[self.triplets[index][2]])
            self.triplet_label_index = [path1, path2,path3]

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return [anchor_img, positive_img, negative_img], [self.triplet_label_index]


    def __len__(self):

       if self.mode == 'train':
           return int(len(self.data['path']))
       else:
           return int(len(self.data['path'])/3)









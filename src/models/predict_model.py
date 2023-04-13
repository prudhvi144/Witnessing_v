import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from util import SiameseNetwork
from libs.dataset import Dataset
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v',
        '--val_path',
        type=str,
        help="Path to directory containing validation dataset.",
        required=False,
        default='../data/external/test_filtered.csv'
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="../reports/siamese/",
        required=False,
        default='../reports/siamese/'
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        help="Path of model checkpoint to be used for inference.",
        required=False,
        default='../models/siamese/resnet/epoch_25.pth'
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_dataset = Dataset(args.val_path, shuffle_pairs=False, augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    criterion = torch.nn.BCELoss()

    checkpoint = torch.load(args.checkpoint)
    model = SiameseNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    losses = []
    correct = 0
    total = 0
    correct_pos = 0
    wrong_pos = 0
    correct_neg = 0
    wrong_neg = 0
    prob_app = []
    lab = []
    inv_transform = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                             std=[1., 1., 1.]),
                                        ])

    for i, ((img1, img2), y, (class1, class2)) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        # print("[{} / {}]".format(i, len(val_dataloader)))

        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        class1 = class1[0]
        class2 = class2[0]
        prob = model(img1, img2)
        loss = criterion(prob, y)

        losses.append(loss.item())
        # Convert prob to binary labels using threshold 0.5
        pred_labels = torch.where(prob > 0.5, torch.ones_like(prob), torch.zeros_like(prob))

        # Compare predicted labels with ground truth labels to get tensor of 1's and 0's
        correct_labels = torch.eq(pred_labels, y)

        # Count number of correct predictions
        correct += torch.sum(correct_labels).item()

        total += len(y)

        if not correct_labels.all():
            # Misclassified image pair, save visualization
            fig = plt.figure("class1={}\tclass2={}".format(class1, class2), figsize=(4, 2))
            plt.suptitle("cls1={}  conf={:.2f}  cls2={}".format(class1, prob[0][0].item(), class2))

            img1 = inv_transform(img1).cpu().numpy()[0]
            img2 = inv_transform(img2).cpu().numpy()[0]

            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(img1[0], cmap=plt.cm.gray)
            plt.axis("off")

            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(img2[0], cmap=plt.cm.gray)
            plt.axis("off")

            plt.savefig(os.path.join(args.out_path, '{}.png').format(i))
    # num_misclassified = total - correct
    # print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t Misclassified={}/{}".format(
    #     sum(losses) / len(losses), correct / total, num_misclassified, total))
    print(correct_pos)
    print(correct)
    print(total)
    print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses) / len(losses), correct / total))
from sys import path
from torch.types import Device
from detection.mtcnn import MTCNN
from classification.facenet.inception_resnet_v1 import InceptionResnetV1, load_weights
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import cv2
from pprint import pprint
import pandas as pd
from utils.general import tensor2numpy, numpy2tensor
import time
import os
import argparse
import sys
from config import get_classification_model, config

parser = argparse.ArgumentParser('Parse dataset constants')
parser.add_argument('--pretrained', action="store", default='VggFace2', help='Classifier model, VggFace2, CelebA')
parser.add_argument('--image', metavar='image', type=str, help='Path to image directory', required='--pretrained' in sys.argv)
parser.add_argument('--save', metavar='-s', help='Save embeddings')
parser.add_argument('--dataset', metavar='dataset', type=str, help='Dataset url', required='--pretrained' in sys.argv)
parser.add_argument('--cls-embed-path', metavar='cls-embed-path', 
                        type=str, 
                        default= config['EMBEDDINGS']['facenet_embeddings_path'],
                        help='Classifier embeddings saving path', 
                        required=False)
parser.add_argument('--cropped-face-save-path', metavar='cropped-face-save-path', 
                        type=str, 
                        default= config['EMBEDDINGS']['aligned_mtcnn_dataset'],
                        help='Classifier embeddings saving path', 
                        required=False)

args = parser.parse_args()
model = args.pretrained
image = args.image # input image url
save_cache = args.save
dataset = args.dataset
loaded = False
facenet_embeddings_path = args.cls_embed_path
aligned_mtcnn_dataset_path = args.cropped_face_save_path

if os.path.isfile(facenet_embeddings_path+'.npy') and os.path.isfile(aligned_mtcnn_dataset_path+'.npy'):
    loaded = True


print(model)
print(image)
print(save_cache)

classifier_weights = get_classification_model(model)

workers = 0 if os.name == 'nt' else 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(custom=classifier_weights, classify=False, num_classes=66).eval().to(device)

def collate_fn(x):
    return x[0]

def register_dataset(path):
    dataset = datasets.ImageFolder(path)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    # print(loader)
    return loader, dataset

def recognize(face:np.ndarray, dataset:str, save_cache:bool=False, n_closest_faces:int=5):
    global facenet_embeddings_path
    global aligned_mtcnn_dataset_path
    
    loader, dataset = register_dataset(dataset)

    distances = []
    aligned = []
    names = []
    for x, y in loader:
        t = time.time()
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            # print(f'type of x_aligned: {type(x_aligned)}')
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])
        print(f'face detection time: {time.time()-t}')
    
    # get stacked faces
    aligned = torch.stack(aligned).to(device)
    
    # detect face in single image
    x_aligned_ground_truth, prob_ground_truth = mtcnn(face, return_prob=True)
    face_aligned = torch.stack([x_aligned_ground_truth]).to(device)

    # compute embeddings
    if save_cache and not loaded:
        embedding_dataset = resnet(aligned).detach() # list of dataset embeddings
        embedding_face = resnet(face_aligned).detach()
        # save embeddings to file
        tensor2numpy(embedding_dataset, facenet_embeddings_path)
        tensor2numpy(embedding_face, aligned_mtcnn_dataset_path)
        embedding_dataset = embedding_dataset.cpu()
        embedding_face = embedding_face.cpu()
        embedding_face = embedding_face[0] # get ground truth face embedding
    elif loaded:
        print(f'Loading tensors from cache')
        embedding_dataset = numpy2tensor(facenet_embeddings_path)
        embedding_face = numpy2tensor(aligned_mtcnn_dataset_path)
    elif not loaded and not save_cache:
        embedding_dataset = resnet(aligned).detach().cpu() # list of dataset embeddings
        embedding_face = resnet(face_aligned).detach().cpu()
        embedding_face = embedding_face[0] # get ground truth face embedding

    # compare face embedding with dataset embeddings
    for dist, name in zip(embedding_dataset, names):
        d = (embedding_face-dist).norm().item()
        distances.append({"distance": d, "class": name})
    
    distances = sorted(distances, key=lambda k: k['distance'])
    # pprint(sorted(distances, key=lambda k: k['distance']))

    return distances[:n_closest_faces]

if __name__ == "__main__":
    
    dist = recognize(face = cv2.imread(image), 
                     dataset=dataset,
                     save_cache=save_cache,
                     n_closest_faces = 5)


    # img = cv2.imread('datasets/test_images/13/096418.jpg')
    # dist = recognize(img, 'datasets/test_images/')
    pprint(dist)
from dataloaders.dataset import VideoDataset
import os
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path

class ProtoNetVideoDataset(VideoDataset):
    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False, n_samples=5):
        super().__init__(dataset=dataset, split=split, clip_len=clip_len, preprocess=preprocess)
        self.n_samples = n_samples

    def __getitem__(self, index):
        # Get the label of the current video
        label = self.label_array[index]

        # Select n_samples random samples with the same label (for the support set)
        support_indices = np.where(self.label_array == label)[0]
        support_indices = np.random.choice(support_indices, self.n_samples, replace=False)

        # Select n_samples random samples with the same label (for the query set),
        # excluding the samples in the support set
        query_indices = np.setdiff1d(np.where(self.label_array == label)[0], support_indices)
        query_indices = np.random.choice(query_indices, self.n_samples, replace=False)

        # Load and preprocess support set
        support_set = []
        for i in support_indices:
            buffer = self.load_frames(self.fnames[i])
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            support_set.append(torch.from_numpy(buffer))

        # Load and preprocess query set
        query_set = []
        for i in query_indices:
            buffer = self.load_frames(self.fnames[i])
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            query_set.append(torch.from_numpy(buffer))

        return support_set, query_set, label

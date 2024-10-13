import torch
import random
from torch.utils.data import Dataset

class CustomConcatDataset(Dataset):
    def __init__(self, dataset_list, dataset_ratio=None, samples_per_epoch=160000):
        self.dataset_list = dataset_list
        if dataset_ratio is not None :
            assert len(dataset_ratio) == len(dataset_list)
        else :
            dataset_ratio = [1] * len(dataset_list)
        self.dataset_ratio = dataset_ratio
        self.samples_per_epoch = samples_per_epoch

    def __len__(self,):
        return self.samples_per_epoch

    def __getitem__(self, index):
        dataset_idx = random.choices(list(range(len(self.dataset_ratio))), weights=self.dataset_ratio, k=1)[0]
        dataset = self.dataset_list[dataset_idx]
        index = random.randint(0, len(dataset) - 1)
        return dataset[index]
import json
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from abc import abstractmethod


# novel
nWorker_setDM = 0
nWorker_labCB = 0
# source
nWorker_simpleDM = 0
nWorker_unlabCB = 0
identity = lambda x: x


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


def list_set_collate(batch):
    # print(f"the batch is {batch}")
    x_batch = torch.Tensor([])
    xlist_batch = []
    y_batch = torch.Tensor([])
    for i, sample in enumerate(batch):
        if type(sample[0]) == tuple:
            # print(f"the sample is {sample}")
            print(f"the sample[0] is {sample[0]}")
            print(f"the sample[1] is {sample[1]}")

        # import pdb; pdb.set_trace() #Ankit
        x_batch = torch.cat([x_batch, sample[0].unsqueeze(0)], dim=0)
        xlist_batch.append(sample[1])
        y = sample[2]
        if isinstance(y, int):
            y = torch.Tensor([y])
        else:
            y = y.unsqueeze(0)
        y_batch = torch.cat([y_batch, y], dim=0)
    return x_batch, xlist_batch, y_batch


class SetDataManager(DataManager):
    def __init__(
        self,
        image_size,
        num_aug,
        n_way,
        n_support,
        n_query,
        n_episode=100,
        no_color=False,
    ):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.n_support = n_support
        self.batch_size = n_support + n_query  ## 20
        self.n_episode = n_episode
        self.num_aug = num_aug
        # self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file):
        # parameters that would change on train/val set

        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(
        #             (self.image_size, self.image_size)
        #         ),  # Specify your desired height and width
        #         transforms.ToTensor(),
        #     ]
        # )  # Ankit

        transform_test = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                # Specify your desired height and width
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-10, 10)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomAffine(degrees=0, shear=(-10, 10)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )  # Ankit

        dataset = SetDataset(
            data_file, self.batch_size, self.num_aug, transform_test
        )  ###Added neagtive file
        sampler = EpisodicBatchSampler(
            len(dataset), self.n_way, self.n_episode
        )  ### 50 classes with 5 way images
        data_loader_params = dict(
            batch_sampler=sampler,
            num_workers=nWorker_setDM,
            collate_fn=list_set_collate,
        )  # collate_func =list_set_collate
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


class SetDataset:
    def __init__(
        self, data_file, batch_size, num_aug, transform_test=transforms.ToTensor()
    ):
        self.data_file = data_file
        with open(data_file, "r") as f:
            self.meta = json.load(f)
        self.cl_list = np.unique(self.meta["image_labels"]).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta["image_names"], self.meta["image_labels"]):
            #   print(f"the image is {x},{y}")
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        # self.sub_negative_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # use main thread only or may receive multiple batches
            pin_memory=False,
            collate_fn=list_set_collate,
        )
        for cl in self.cl_list:
            sub_dataset = SubDataset(
                self.sub_meta[cl], cl, num_aug=num_aug, transform_test=transform_test
            )  ###
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)
            )

    def __getitem__(self, i):
        clswise_list = next(iter(self.sub_dataloader[i]))

        return clswise_list

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(
        self,
        sub_meta,
        cl,
        num_aug=100,
        transform_test=transforms.ToTensor(),
        target_transform=identity,
        min_size=50,
    ):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform_test = transform_test
        self.target_transform = target_transform
        if len(self.sub_meta) < min_size:
            idxs = [i % len(self.sub_meta) for i in range(min_size)]
            self.sub_meta = np.array(self.sub_meta)[idxs].tolist()

    def __getitem__(self, i):
        image_path = self.sub_meta[i]
        img_raw = Image.open(image_path).convert("RGB")
        img = self.transform_test(img_raw)
        target = self.target_transform(self.cl)
        return img, img_raw, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[: self.n_way]
            # yield torch.tensor([5,2,0,4,6])

# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple

import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS, DatasetFolder
from torchvision.transforms import transforms

try:
    from torchvision.transforms import InterpolationMode

    interpolation = InterpolationMode.BICUBIC
except:
    import PIL

    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img: PImage.Image = PImage.open(f).convert("RGB")
    return img


class ImageNetDataset(DatasetFolder):
    def __init__(
        self,
        imagenet_folder: str,
        train: bool,
        transform: Callable,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, "train" if train else "val")
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None,
            is_valid_file=is_valid_file,
        )

        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None  # this is self-supervised learning so we don't need labels

    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))


# ############################# MY CODE #############################
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import torch
import torchvision.transforms as T


class PretrainingDatasetSparK:
    def __init__(
        self,
        dataset: str,
        in_channels: int = 1,
        image_shape: Tuple = [768, 1024],
        unlabelled: str = "bw",
        trans_train: Optional[Callable] = None,
    ):
        assert dataset in ["4k", "test"], "The dataset can be only 4k or test"
        assert unlabelled in ["bw", "bw+"], "Unlabelled can be None or one between ['bw', 'bw+]"
        self.in_channels = in_channels

        data4k = Path("/cluster/group/karies_2022/Data/dataset_4k/images/")
        unlabelled_bw = Path("/cluster/group/karies_2022/Data/unlabelled_images/BW für FH Jan24")
        einzelzahnbilder = Path("/cluster/group/karies_2022/Data/unlabelled_images/Einzelzahnbilder")

        self.img_paths = []

        for paket in data4k.glob("*"):
            self.img_paths.extend(list((data4k / paket).glob("*")))
            if dataset == "test":
                break

        if unlabelled == "bw+":
            self.img_paths.extend(list(einzelzahnbilder.glob("*")))
        if unlabelled in ["bw", "bw+"]:
            self.img_paths.extend(list(unlabelled_bw.glob("*")))

        image_transform = [T.ToTensor()]
        if image_shape is not None:
            image_transform.append(T.Resize((image_shape[0], image_shape[1]), antialias=True))

        image_transform.extend(trans_train)
        self.image_transform = T.Compose(image_transform)

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get item from dataset.

        Args:
            index (int): index of element

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: image tensor and label dict with boxes, labels and masks
        """

        image = cv2.imread(str(self.img_paths[index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.image_transform(image)

        if self.in_channels == 1:
            image_tensor = image_tensor[0:1]
        return image_tensor


# ##################################################################


def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow.

    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    input_size = int(input_size)
    trans_train = [
        transforms.RandomResizedCrop((input_size, input_size), scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]

    # dataset_path has to be either "4k" or "test", and that is set in the args
    image_shape = [768, 1024]
    unlabelled = "bw"
    in_channels = 1
    dataset_train = PretrainingDatasetSparK(
        dataset=dataset_path,
        image_shape=image_shape,
        unlabelled=unlabelled,
        trans_train=trans_train,
        in_channels=in_channels,
    )

    print_transform(T.Compose(trans_train), "[pre-train]")
    return dataset_train


def print_transform(transform, s):
    print(f"Transform {s} = ")
    for t in transform.transforms:
        print(t)
    print("---------------------------\n")

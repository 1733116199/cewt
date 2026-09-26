# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
import torchvision, PIL.Image, pickle
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class IM100Dataset(datasets.ImageFolder):
    def __init__(self, root, transform = None):
        super().__init__(root, transform=transform)
        self.full_imnet_class_to_idx = self.find_imnet_classes(root)[1]
        self.my_ids = []
        for c in self.classes:
            self.my_ids.append(self.full_imnet_class_to_idx[c])
    
    def find_imnet_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def find_classes(self, directory):
        # https://github.com/danielchyeh/ImageNet-100-Pytorch/blob/5ae5a42f74a23e8107aa060067fc355f16007d91/IN100.txt
        with open("IN100.txt", "r") as f:
            classes = sorted([line.replace('\n', '') for line in f.readlines()])
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def _find_classes(self, directory):
        # https://github.com/danielchyeh/ImageNet-100-Pytorch/blob/5ae5a42f74a23e8107aa060067fc355f16007d91/IN100.txt
        with open("IN100.txt", "r") as f:
            classes = sorted([line.replace('\n', '') for line in f.readlines()])
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

IMNET64_DEFAULT_MEAN = [0.481, 0.458 , 0.408]
IMNET64_DEFAULT_STD = [0.269, 0.261, 0.276]
class IMNET64(torchvision.datasets.vision.VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        import numpy as np
        img_size = 64
        img_size2 = img_size * img_size
        if self.train:
            filenames = [f"{root}/train_data_batch_{i}" for i in range(1, 11)]
        else:
            filenames = [f"{root}/val_data"]

        xs = []
        ys = []
        for filename in filenames:
            print(f"Loading {filename}...")
            with open(filename, "rb") as f:
                entry = pickle.load(f)
                x = entry["data"]
                y = entry["labels"]
                x = np.dstack((
                    x[:, :img_size2], 
                    x[:, img_size2:2*img_size2], 
                    x[:, 2*img_size2:]
                ))
                x = x.reshape((x.shape[0], img_size, img_size, 3))
                xs.append(x)
                ys.append(np.asarray(y) - 1)

        self.xs = xs
        self.ys = ys
        self.length = sum([len(xs) for xs in self.xs])

    def __getitem__(self, index: int):
        while index < 0:
            index += self.length
        
        file = 0
        while index >= len(self.xs[file]):
            index -= len(self.xs[file])
            file += 1
        
        img, target = self.xs[file][index], self.ys[file][index]

        img = PIL.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    print(f"transform: {is_train}\n", transform)
    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET100':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = IM100Dataset(root, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET64':
        dataset = IMNET64(args.data_path, train=is_train, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN if args.data_set != "IMNET64" else IMNET64_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD if args.data_set != "IMNET64" else IMNET64_DEFAULT_STD
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            no_aug=args.no_aug,
            mean=mean,
            std=std,
            scale=(args.scale_min, 1.0),
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

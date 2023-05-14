from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from .base_data_loader import BaseDataLoader
from .dataset_caltech import Caltech256
from .dataset_flowers import OxfordFlowers102Dataset
from .dataset_aircrafts import AircraftDataset
from .dataset_birds import BirdDataset
from .dataset_cars import CarDataset
from .dataset_dogs import DogDataset
from .dataset_indoor import IndoorDataset
from .dataset_cifar import CIFAR10, CIFAR100
from .dataset_messidor2 import Messidor2
from .dataset_aptos import Aptos
from .dataset_jinchi import Jinchi
from .mini_domain_net import DomainNetDataLoader
from .animal_attributes import AnimalAttributesDataLoader
from .rand_augment import TransformFixMatch
from .cxr_dataset import CXRDataset
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np

class MatchChannel(object):
    def __call__(self, pic):
        if pic.size()[0] == 1:
            assert len(pic.size()) == 3
            pic = pic.repeat(3,1,1)
        return pic

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split=0, num_workers=num_workers)

class Cifar10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, num_workers=1, phase="train", **kwargs):
        training = phase == "train"
        if training:
            trsfm = BaseDataLoader.train_transform
        else:
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class Cifar100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, num_workers=1, phase="train", **kwargs):
        training = phase == "train"
        if training:
            trsfm = BaseDataLoader.train_transform
        else:
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class CaltechDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, idx_start, img_num, num_workers, phase = "train", **kwargs):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = Caltech256(
            self.data_dir, transform=trsfm, download=True, idx_start = idx_start, img_num=img_num
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=0, test_split=0, num_workers=num_workers)

class FlowerDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, valid_split, test_split, num_workers, **kwargs):
        trsfm = BaseDataLoader.train_transform
        self.data_dir = data_dir
        self.dataset = OxfordFlowers102Dataset(
            root_dir = self.data_dir, transform=trsfm, download=True 
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split, test_split, num_workers)

class AircraftsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, phase = "train", **kwargs):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = AircraftDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=0, test_split=0, num_workers=num_workers)

class BirdsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, valid_split = 0, phase = "train", **kwargs):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = BirdDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class CarsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, valid_split = 0, phase = "train", **kwargs):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = CarDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class DogsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, valid_split = 0, phase = "train", **kwargs):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = DogDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class IndoorDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, valid_split = 0, phase = "train", strong_augment = False, **kwargs):
        if phase == "train":
            if strong_augment:
                trsfm = TransformFixMatch(size=224)
            else:
                trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = IndoorDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)



class MessidorDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, test_split=0.0, num_workers=1, phase="train", **kwargs):
        training = phase == "train"
        if phase == "train":
            trsfm = transforms.Compose([
                    transforms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                    transforms.Pad((0, 37, 0, 38)),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
        elif phase == "val" or phase == "test":
            trsfm = transforms.Compose([
                    transforms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                    transforms.Pad((0, 37, 0, 38)),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])

        self.data_dir = data_dir
        self.dataset = Messidor2(self.data_dir, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=test_split, num_workers=num_workers)

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = transforms.Compose([
                    transforms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                    transforms.Pad((0, 37, 0, 38)),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
            return DataLoader(sampler=self.valid_sampler, **kwargs)
        
    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = transforms.Compose([
                    transforms.Resize(224 - 1, max_size=224),  #resizes (H,W) to (149, 224)
                    transforms.Pad((0, 37, 0, 38)),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
            return DataLoader(sampler=self.test_sampler, **kwargs)

class AptosDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, test_split=0.0, num_workers=1, phase="train", **kwargs):
        training = phase == "train"
        if phase == "train":
            trsfm = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
        elif phase == "val" or phase == "test":
            trsfm = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
        
        self.data_dir = data_dir
        self.dataset = Aptos(self.data_dir, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=test_split, num_workers=num_workers)

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
            return DataLoader(sampler=self.valid_sampler, **kwargs)
    
    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
            return DataLoader(sampler=self.test_sampler, **kwargs)

class JinchiDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, test_split=0.0, num_workers=1, phase="train", **kwargs):
        training = phase == "train"
        if phase == "train":
            trsfm = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
        elif phase == "val" or phase == "test":
            trsfm = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
        
        self.data_dir = data_dir
        self.dataset = Jinchi(self.data_dir, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=test_split, num_workers=num_workers)

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
            return DataLoader(sampler=self.valid_sampler, **kwargs)
        
    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                ])
            return DataLoader(sampler=self.test_sampler, **kwargs)
        
class CXRDataLoader(BaseDataLoader):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, test_split=0.0, num_workers=1, phase="train", **kwargs):
        
        if phase == "train":
            trsfm = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.Scale(224),
                    # because scale doesn't always give 224 x 224, this ensures 224 x 224
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(CXRDataLoader.mean, CXRDataLoader.std)
                ])
        elif phase == "val" or phase == "test":
            trsfm = transforms.Compose([
                transforms.Scale(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(CXRDataLoader.mean, CXRDataLoader.std)
            ])
        
        self.data_dir = data_dir
        self.dataset = CXRDataset(self.data_dir, fold=phase, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=test_split, num_workers=num_workers)
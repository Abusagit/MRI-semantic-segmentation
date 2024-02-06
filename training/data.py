import torch

import sys

sys.path.append("../")
from pathlib import Path
import torchvision.transforms as T


from typing import Optional

from torch.utils.data import Dataset


    
    

class MRIDataset(Dataset):
    
    def __init__(self, root: Path, split_mode: str, 
                 transform: Optional[T.Compose]=None, 
                 target_transform: Optional[T.Compose]=None,
                 **kwargs) -> None:
        super().__init__()
        
        self.split_dir = root / split_mode
        
        self._filenames = list(sorted([path.name for path in self.split_dir.glob("*.pt")]))
        
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self) -> int:
        
        return len(self._filenames)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        
        filename = self._filenames[index]
        
        image_and_mask_tensor: torch.Tensor = torch.load(self.split_dir / filename).float()
        
        image, mask = image_and_mask_tensor
        
        return image, mask
    

# Create a dataset wrapper that allows us to perform custom image augmentations
# on both the target and label (segmentation mask) images.
#
# These custom image augmentations are needed since we want to perform
# transforms such as:
# 1. Random horizontal flip
# 2. Image resize
#
# and these operations need to be applied consistently to both the input
# image as well as the segmentation mask.
class MRIDatasetAugmented(MRIDataset):
    def __init__(
        self,
        root: Path,
        split_mode: str,
        target_types="segmentation",
        download=False,
        pre_transform=None,
        post_transform=None,
        pre_target_transform=None,
        post_target_transform=None,
        common_transform=None,
    ):
        super().__init__(
            root=root,
            split_mode=split_mode,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)
        
        # Common transforms are performed on both the input and the labels
        # by creating a 2 channel image and running the transform on both.
        # Then the segmentation mask (2nd channel) is separated out.
        if self.common_transform is not None:
            both = torch.stack([input, target], dim=0)
            both = self.common_transform(both)
            # print(f"{both.shape=}")
            
            input, target = torch.split(both, split_size_or_sections=1, dim=0)
            # print(f"{input.shape=} {target.shape=}")
        
        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)
            
            
        # input = input.view(1, input.shape[1], -1).expand(3, -1, -1)

            
            
        # print(f"!{input.shape=} {target.shape=}")

        return (input, target)
    

# Simple torchvision compatible transform to send an input tensor
# to a pre-specified device.
class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"
    
    
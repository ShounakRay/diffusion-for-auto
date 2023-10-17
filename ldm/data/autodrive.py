import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AUTOBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 horizontal_size=None,
                 vertical_size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.horizontal_size = horizontal_size
        self.vertical_size = vertical_size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        # !NOTE: 256x256 center crop is forced here for VAE consistency (not required idk)
        # TODO: Remove assertions
        # CENTER CROP
        # crop = min(img.shape[0], img.shape[1])
        crop = 256
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img)
        
        # width, height = image.size   # Get dimensions
        # assert width == 320 and height == 320, f"Image is not 480x320. Received: width, height={(width, height)}"

        # if self.horizontal_size is not None or self.vertical_size is not None:
        #     # SQUISH RESIZE
        #     # image = image.resize((self.size, self.size), resample=self.interpolation)

        #     # NON-BALANCED CENTER CROP
        #     # Based on: https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
        #     width, height = image.size   # Get dimensions
        #     assert width == 480 and height == 320, "Image is not 480x320"

        #     left = np.floor((width - self.horizontal_size)/2)
        #     top = np.floor((height - self.vertical_size)/2)
        #     right = np.floor((width + self.horizontal_size)/2)
        #     bottom = np.floor((height + self.vertical_size)/2)

        #     image = image.crop((left, top, right, bottom))

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        
        return example

"""
cd $GROUP_SCRATCH/shounak_files/DATASETS/waymo
find train -maxdepth 1 -type f -name "*.jpg" -print0 | xargs -0 -I {} basename {} > $HOME/diffusion-for-auto/data/autodrive/waymo/waymo_train.txt
"""
class AUTOWaymoTrain(AUTOBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/autodrive/waymo/waymo_train.txt",
                         data_root="/scratch/groups/mykel/shounak_files/DATASETS/waymo/train",
                         **kwargs)

"""
cd $GROUP_SCRATCH/shounak_files/DATASETS/waymo
find test -maxdepth 1 -type f -name "*.jpg" -print0 | xargs -0 -I {} basename {} > $HOME/diffusion-for-auto/data/autodrive/waymo/waymo_val.txt
"""
class AUTOWaymoValidation(AUTOBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/autodrive/waymo/waymo_val.txt",
                         data_root="/scratch/groups/mykel/shounak_files/DATASETS/waymo/test",
                         flip_p=flip_p,
                         **kwargs)

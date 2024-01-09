import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CROP_SQUARE_SIZE = 512

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
        crop = CROP_SQUARE_SIZE
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

# FIXME: [SHOUNAK]
# Note: The following classes are referenced in the config files,
#       .YAML for the CUSTOM-ldm model.

##########################
######### WAYMO ##########
##########################

ABSOLUTE_GROUP_SCRATCH = "/scratch/groups/mykel"
RELATIVE_TEXT_FILE_BASE = "data/autodrive"
WAYMO_CORE = "waymo"
NUIMAGES_CORE = "nuimages"
AUDI_CORE = "audi"
AUDI_512_CORE = "audi-512"
WAYMO_NUIMG_CORE = f"{WAYMO_CORE}_{NUIMAGES_CORE}"

"""
cd $GROUP_SCRATCH/shounak_files/DATASETS/waymo
find train -maxdepth 1 -type f -name "*.jpg" -print0 | xargs -0 -I {} basename {} > $HOME/diffusion-for-auto/{RELATIVE_TEXT_FILE_BASE}/waymo/waymo_train.txt
"""
class AUTOWaymoTrain(AUTOBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/waymo/waymo_train.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/waymo/train",
                         **kwargs)

"""
cd $GROUP_SCRATCH/shounak_files/DATASETS/waymo
find test -maxdepth 1 -type f -name "*.jpg" -print0 | xargs -0 -I {} basename {} > $HOME/diffusion-for-auto/data/autodrive/waymo/waymo_val.txt
"""
class AUTOWaymoValidation(AUTOBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/waymo/waymo_val.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/waymo/test",
                         flip_p=flip_p,
                         **kwargs)

#########################
########## AUDI #########
#########################
"""
cd $GROUP_SCRATCH/audi
find train -maxdepth 1 -type f -name "*.png" -print0 | xargs -0 -I {} basename {} > $HOME/diffusion-for-auto/{RELATIVE_TEXT_FILE_BASE}/audi/audi_train.txt
"""
class AUTOAudiTrain(AUTOBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/{AUDI_CORE}/{AUDI_CORE}_train.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/{AUDI_CORE}/train",
                         **kwargs)

"""
cd $GROUP_SCRATCH/audi
find test -maxdepth 1 -type f -name "*.png" -print0 | xargs -0 -I {} basename {} > $HOME/diffusion-for-auto/{RELATIVE_TEXT_FILE_BASE}/audi/audi_val.txt
"""
class AUTOAudiValidation(AUTOBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/{AUDI_CORE}/{AUDI_CORE}_val.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/{AUDI_CORE}/test",
                         flip_p=flip_p,
                         **kwargs)
        
#############################
########## AUDI-512 #########
#############################
"""
cd $GROUP_SCRATCH/audi-512
find train -maxdepth 1 -type f -name "*.png" -print0 | xargs -0 -I {} basename {} > $HOME/diffusion-for-auto/$RELATIVE_TEXT_FILE_BASE/audi-512/audi_train.txt
"""
class AUTOAudiTrain_512(AUTOBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/{AUDI_512_CORE}/{AUDI_512_CORE}_train.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/{AUDI_512_CORE}/train",
                         **kwargs)

"""
cd $GROUP_SCRATCH/audi-512
find test -maxdepth 1 -type f -name "*.png" -print0 | xargs -0 -I {} basename {} > $HOME/diffusion-for-auto/$RELATIVE_TEXT_FILE_BASE/audi-512/audi_val.txt
"""
class AUTOAudiValidation_512(AUTOBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/{AUDI_512_CORE}/{AUDI_512_CORE}_val.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/{AUDI_512_CORE}/test",
                         flip_p=flip_p,
                         **kwargs)

#########################
####### NUIMAGES ########
#########################
class AUTONuimagesTrain(AUTOBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/{NUIMAGES_CORE}/{NUIMAGES_CORE}_train.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/{NUIMAGES_CORE}/train",
                         **kwargs)

class AUTONuimagesValidation(AUTOBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/{NUIMAGES_CORE}/{NUIMAGES_CORE}_val.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/{NUIMAGES_CORE}/test",
                         flip_p=flip_p,
                         **kwargs)
        
#########################
### WAYMO + NUIMAGES ####
#########################

"""
Steps used to generate:
*** NOTE TRAILING SLASHES ***
*** REFERENCE: https://serverfault.com/a/505758 ***

1. Local joint data, test directory for waymo
`
time rsync -avhW --no-compress \
    $GROUP_SCRATCH/shounak_files/DATASETS/waymo/test/ \
    $GROUP_SCRATCH/shounak_files/DATASETS/waymo_nuimages/test
`

2. Local joint data, test directory for nuimages
`
time rsync -avhW --no-compress \
    $GROUP_SCRATCH/shounak_files/DATASETS/nuimages/test/ \
    $GROUP_SCRATCH/shounak_files/DATASETS/waymo_nuimages/test
`

3. Local joint data, train directory for waymo
`
time rsync -avhW --no-compress \
    $GROUP_SCRATCH/shounak_files/DATASETS/waymo/train/ \
    $GROUP_SCRATCH/shounak_files/DATASETS/waymo_nuimages/train
`

4. Local joint data, train directory for nuimages
`
time rsync -avhW --no-compress \
    $GROUP_SCRATCH/shounak_files/DATASETS/nuimages/train/ \
    $GROUP_SCRATCH/shounak_files/DATASETS/waymo_nuimages/train
`

5. Generate joint test file
ls test/ -1 > $HOME/diffusion-for-auto/data/autodrive/waymo_nuimages/waymo_nuimages_val.txt

6. Generate joint train file
ls train/ -1 > $HOME/diffusion-for-auto/data/autodrive/waymo_nuimages/waymo_nuimages_train.txt

"""

class AUTOWaymoNuimagesTrain(AUTOBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/{WAYMO_NUIMG_CORE}/{WAYMO_NUIMG_CORE}_train.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/{WAYMO_NUIMG_CORE}/train",
                         **kwargs)

class AUTOWaymoNuimagesValidation(AUTOBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file=f"{RELATIVE_TEXT_FILE_BASE}/{WAYMO_NUIMG_CORE}/{WAYMO_NUIMG_CORE}_val.txt",
                         data_root=f"{ABSOLUTE_GROUP_SCRATCH}/shounak_files/DATASETS/{WAYMO_NUIMG_CORE}/test",
                         flip_p=flip_p,
                         **kwargs)
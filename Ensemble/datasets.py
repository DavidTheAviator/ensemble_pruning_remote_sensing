import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



##############
##  LANDCOVER
##############
#region

#Source  https://github.com/milesial/Pytorch-UNet/blob/master/utils/data_loading.py
#Adapted (Made the Class fit my folder structure; implemented transform/normalizing logic)
class LandcoverDataset(Dataset):
    """
    Custom Dataset class for loading images and masks for the Landcover Dataset

    Args:
        images_dir (str): Path to the directory containing image files
        mask_dir (str): Path to the directory containing mask files
        indexing_file (str): Path to a file containing a list of image IDs
        mask_suffix (str): Suffix added to the image ID to match the mask file
        transform (list, optional): A list of image augmentation transforms to apply to the images
        normalizer (callable, optional): A callable to normalize the image tensors


    Methods:
        __len__(): Returns the total number of images in the dataset
        __getitem__(idx): Loads and returns the image and mask pair at the specified index (optionally with transforms/normalization)
    """


  
    def __init__(self, images_dir: str, mask_dir: str, indexing_file: str, mask_suffix: str, transform = None, normalizer = None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.indexing_file = Path(indexing_file)
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.normalizer = normalizer



        self.ids = None
        with open(self.indexing_file, 'r') as file:
          lines = [line.rstrip() for line in file]
          self.ids = lines


        if not self.ids:
            raise RuntimeError(f'There were no files provided in  {indexing_file}, it should contain a list of image indeces.')

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'



        #Image now has format:  Height x Width x N_channels
        img = np.asarray(Image.open(img_file[0]))

        #convert to N_channels x Height x Width
        img = np.moveaxis(img,-1,0)

        #Mask has format: Height x Width x N_channels
        mask = np.asarray(Image.open(mask_file[0]))

        #get rid of the two unnecessary channels and format to: Height x Width
        mask = np.squeeze(mask[:,:,:1])

        assert img.shape[1:] == mask.shape[:], \
            f'Image and mask {name} should have the same height and width, but are {img.shape} and {mask.shape}'


        #check for normalizer
        if self.normalizer is None:
          if(self.transform is not None):
            raise ValueError('No normalizer, but a transform provided! Provide normalizer and transform!')
          
          return {
              'image': torch.as_tensor(img.copy()).float().contiguous(),
              'mask': torch.as_tensor(mask.copy()).long().contiguous()
          }


        #transform and normalize
        elif self.transform is not None:
          #choose one element from the transform list
          trans_ind = np.random.randint(len(self.transform))



          #create the transform for the image
          transform_norm = v2.Compose([
              v2.ToDtype(torch.uint8),
              self.transform[trans_ind],
              v2.ToDtype(torch.float32),
              self.normalizer])

          #use tv_tensors to make v2 transforms work
          image = tv_tensors.Image(torch.as_tensor(img.copy()))
          mask = tv_tensors.Image(torch.as_tensor(mask.copy()))




          #check if there is a transform for the mask needed
          trans_name = type(self.transform[trans_ind]).__name__
          if  trans_name == 'RandomHorizontalFlip' or trans_name == 'RandomVerticalFlip':
            trans_flip = v2.Compose([
                v2.ToDtype(torch.uint8),
                self.transform[trans_ind],
                v2.ToDtype(torch.float32)
            ])




            image, mask = trans_flip(image, mask)

            return {
                'image': self.normalizer(image).float().contiguous(),
                'mask': torch.squeeze(mask[0,:,:]).long().contiguous()
            }

          #the image is not flipped, we can transform just the image
          else:
            return {
              'image': transform_norm(image).float().contiguous(),
              'mask': torch.squeeze(mask[0,:,:]).long().contiguous()
            }


        #just normalize
        else:
          return {
              'image': self.normalizer(torch.as_tensor(img.copy()).float()).contiguous(),
              'mask': torch.as_tensor(mask.copy()).long().contiguous()

          }
#endregion        




##############
##  FLOODS
##############
#region

#Source  https://github.com/milesial/Pytorch-UNet/blob/master/utils/data_loading.py
#Adapted (Made the Class fit my folder structure; implemented transform/normalizing logic)
class FloodDataset(Dataset):
    """
    Custom Dataset class for loading images and masks for the S1GFlood Dataset

    Args:
        data_dir (str): Path to the directory containing (pre and post) image and mask files
        index_file (str): Name of the index file, which should be inside the data directory
        transform (list, optional): A list of image augmentation transforms to apply to the images
        normalizer_pre (callable, optional): A callable to normalize the (pre flood) image tensors
        normalizer_post (callable, optional): A callable to normalize the (post flood) image tensors


    Methods:
        __len__(): Returns the total number of images in the dataset
        __getitem__(idx): Loads and returns the pre and post flood image and the mask at the specified index (optionally with transforms/normalization)
    """
    def __init__(self, data_dir: str,index_file:str, transform = None, normalizer_pre = None, normalizer_post = None):
        self.images_pre_dir = Path(data_dir).joinpath('A')
        self.images_post_dir = Path(data_dir).joinpath('B')
        self.mask_dir = Path(data_dir).joinpath('label')
        self.transform = transform
        self.normalizer_pre = normalizer_pre
        self.normalizer_post = normalizer_post
        self.index_file = Path(data_dir).joinpath(index_file)




        #save the file names as ids
        with open(self.index_file, 'r') as file:
          lines = [line.rstrip() for line in file]
          self.ids = lines


        if not self.ids:
            raise RuntimeError(f'Error in reading out the file names from  {self.mask_dir}')

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        name = self.ids[idx]
        img_pre_file = list(self.images_pre_dir.glob(name ))
        img_post_file = list(self.images_post_dir.glob(name ))
        mask_file = list(self.mask_dir.glob(name))



        assert len(img_pre_file) == 1, f'Either no pre image or multiple pre images found for the ID {name}: {img_pre_file}'
        assert len(img_post_file) == 1, f'Either no post image or multiple post images found for the ID {name}: {img_post_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'



        #Image now has format:  Height x Width x N_channels
        img_pre = np.asarray(Image.open(img_pre_file[0]))
        img_post = np.asarray(Image.open(img_post_file[0]))

        #convert to N_channels x Height x Width
        img_pre = np.moveaxis(img_pre,-1,0)
        img_post = np.moveaxis(img_post,-1,0)





        #Mask has format: Height x Width
        mask = np.asarray(Image.open(mask_file[0]))




        assert img_pre.shape[1:] == mask.shape[:], \
            f'Image and mask {name} should have the same height and width, but are {img_pre.shape} and {mask.shape}'
        assert img_pre.shape == img_post.shape, \
            f'Pre and Post Image {name} should have the same shape, but have shape: {img_pre.shape} and {img_post.shape}'



        assert self.normalizer_pre is None and self.normalizer_post is None or self.normalizer_pre is not None and self.normalizer_post is not None, \
            f'Provide either both normalizers or zero!'


        #check for normalizer
        if self.normalizer_pre is None:
          if(self.transform is not None):
            raise ValueError('No normalizer, but a transform provided! Provide normalizer and transform!')
          return {
              'image_pre': torch.as_tensor(img_pre.copy()).float().contiguous(),
              'image_post': torch.as_tensor(img_post.copy()).float().contiguous(),
              'mask': torch.as_tensor(mask.copy()).long().contiguous()
          }


        #transform and normalize
        elif self.transform is not None:
          #choose one element from the transform list
          trans_ind = np.random.randint(len(self.transform))


          #create the transform for the image (and potentially the mask)
          transform = v2.Compose([
              v2.ToDtype(torch.uint8),
              self.transform[trans_ind],
              v2.ToDtype(torch.float32)])


          #use tv_tensors to make v2 transforms work
          img_pre = tv_tensors.Image(torch.as_tensor(img_pre.copy()))
          img_post = tv_tensors.Image(torch.as_tensor(img_post.copy()))
          mask = tv_tensors.Image(torch.as_tensor(mask.copy()))


          #transform all at once (otherwise some images, get flipped while others don't)
          img_pre_trans, img_post_trans, mask_trans = transform(img_pre,img_post,mask)


          #check if there is a transform for the mask needed
          trans_name = type(self.transform[trans_ind]).__name__

          if  trans_name == 'RandomHorizontalFlip' or trans_name == 'RandomVerticalFlip':

            return {
                'image_pre': self.normalizer_pre(img_pre_trans).float().contiguous(),
                'image_post': self.normalizer_post(img_post_trans).float().contiguous(),
                'mask': torch.squeeze(mask_trans[0,:,:]).long().contiguous()
            }



          #the image is not flipped, we can transform just the image
          else:
            return {
                'image_pre': self.normalizer_pre(img_pre_trans).float().contiguous(),
                'image_post': self.normalizer_post(img_post_trans).float().contiguous(),
                'mask': torch.squeeze(mask[0,:,:]).long().contiguous()
            }


        #just normalize
        else:
          return {
              'image_pre': self.normalizer_pre(torch.as_tensor(img_pre.copy()).float()).contiguous(),
              'image_post': self.normalizer_post(torch.as_tensor(img_post.copy()).float()).contiguous(),
              'mask': torch.as_tensor(mask.copy()).long().contiguous()

          }
        

#endregion

from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path, max_retry=3):
    if not osp.exists(img_path):
        print("[Warning] {} does not exist. Skipping.".format(img_path))
        return None
    
    retry_count = 0
    while retry_count < max_retry:
        try:
            with Image.open(img_path) as img:
                return img.convert('RGB')
        except IOError:
            retry_count += 1
            print("[Warning] Retry {}/{} for reading '{}'...".format(retry_count, max_retry, img_path))
    
    # 读取失败，直接返回 None
    print("[Warning] Failed to read '{}'. Skipping.".format(img_path))
    return None




class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  -----------------------")
        print("  subset   | # images")
        print("  -----------------------")
        print("  train    | {:8d}".format(num_train_imgs))
        print("  query    | {:8d}".format(num_query_imgs))
        print("  gallery  | {:8d}".format(num_gallery_imgs))
        print("  -----------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid,img_path.split('/')[-1]
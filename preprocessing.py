import os
import glob
import numpy as np
import random

from torch.utils.data import Dataset

import utils
import pickle

import albumentations as A

class img_dataset(Dataset):
    def __init__(self,data_dir, transform_shape = None, transform_color = None, sample = False, sample_number = 4,
                 sample_anomaly = None,  sample_cond_threshold = 0, slice_offset = 20):
        """
        Args:
        :param data_dir: Directory of images (str)
        :param transform_shape: Albumentations transforms
        :param transform_color: Albumentations transforms
        :param sample: Return sampled slices
        :param sample: Number of slices to be sampled
        :param sample_anomally: [None, normal, abnormal]
        :param sample_cond_threshold: Threshold to apply to the label so define anomaly (e.g. labels in segmentation > 1 are anomalies)
        """
        self.data_dir = data_dir
        self.set = glob.glob(data_dir+'/*.nt')
        self.transform_shape = transform_shape
        self.transform_color = transform_color
        self.sample = sample
        self.sample_number = sample_number
        self.sample_anomaly = sample_anomaly
        self.slice_offset = slice_offset
        self.sample_cond_threshold = sample_cond_threshold

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        file_name = os.path.join(self.data_dir,self.set[item])
        img_batch = pickle.load(open(file_name, 'rb'))

        if self.sample:
            sampled_img_batch = []
            for img_ext in img_batch:

                if self.sample_anomaly == 'normal':
                    msk_normal = ~np.any(img_ext.seg > self.sample_cond_threshold, axis=(1,2))
                    msk_normal[:self.slice_offset] = False
                    msk_normal[-self.slice_offset:] = False
                    msk_normal = msk_normal & (~np.all(img_ext.img == 0,axis=(1,2))) # Remove empty planes
                    choices = np.arange(len(msk_normal))[msk_normal]

                elif self.sample_anomaly == 'abnormal':
                    msk_abnormal = np.any(img_ext.seg > self.sample_cond_threshold, axis=(1,2))
                    msk_abnormal[:self.slice_offset] = False
                    msk_abnormal[-self.slice_offset:] = False
                    msk_abnormal = msk_abnormal & (~np.all(img_ext.img == 0,axis=(1,2))) # Remove empty planes
                    choices = np.arange(len(msk_abnormal))[msk_abnormal]

                elif self.sample_anomaly is None:
                    msk_empty = ~np.all(img_ext.img == 0,axis=(1,2)) # Remove empty planes
                    msk_empty[:self.slice_offset] = False
                    msk_empty[-self.slice_offset:] = False
                    choices = np.arange(len(msk_empty))[msk_empty]
                else:
                    raise NotImplementedError

                # np.random does not work inside the dataloader
                sample_idx = np.array(random.choices(choices,k = self.sample_number))
                img = img_ext.img[sample_idx].astype(np.float32)
                seg = img_ext.seg[sample_idx] if img_ext.seg is not None else np.zeros_like(img).astype('uint8')

                # Normalize using full volume statistics
                non_cero_mask = np.where(img_ext.img > 0.05)
                mu, std = img_ext.img[non_cero_mask].mean(),img_ext.img[non_cero_mask].std()

                non_cero_mask = np.where(img > 0.05)
                img[non_cero_mask] = (img[non_cero_mask] - mu)/std

                # Coordinates in range [-.5,.5] for conditionning
                coord = sample_idx[:, np.newaxis] / img_ext.img.shape[0]
                coord = coord - 0.5

                sampled_img_batch.append(utils.img_extended(img,\
                                                            seg,\
                                                            np.array([None]*self.sample_number),\
                                                            np.any(seg > self.sample_cond_threshold,axis=(1,2)),
                                                            coord,
                                                            np.array([img_ext.cid]*self.sample_number)))

            img_batch = collate_fn(sampled_img_batch)


        else: # If no ssampling is required, just reverse the order: list of img_ext to img_ext of arrays
            img_batch = utils.img_extended(*map(np.array, zip(*img_batch)))

        if self.transform_shape is not None:
            img_aug = []
            seg_aug = []
            for img,seg in zip(img_batch.img, img_batch.seg):
                tmp = self.transform_shape(image = img, mask = seg)
                img_aug.append(tmp['image'])
                seg_aug.append(tmp['mask'])

            img_batch = utils.img_extended(np.stack(img_aug), np.stack(seg_aug), img_batch.k,
                                           img_batch.t, img_batch.coord, img_batch.cid)

        if self.transform_color is not None:
            cero_mask = img_batch.img == 0
            # Set to range [0,1], clipping any value further than 3 sigma
            img_aug = np.clip((img_batch.img + 3.) / 6., 0, 1)
            img_aug = np.stack([self.transform_color(image=i)['image'] for i in img_aug])
            img_aug = img_aug * 6. - 3.
            img_aug[cero_mask] = 0

            img_batch = utils.img_extended(img_aug, img_batch.seg, img_batch.k,
                                           img_batch.t, img_batch.coord, img_batch.cid)

        return img_batch

def collate_fn(batches):
    batch = utils.img_extended(*map(np.concatenate, zip(*batches)))
    return batch

class brain_dataset(img_dataset):
    def __init__(self, data_dir, train = True,  **kwargs):

        if train:
            transform_shape = A.Compose([A.ElasticTransform(alpha = 2, sigma = 5, alpha_affine = 5),
                                        A.RandomScale((-.15, .1)),
                                        A.PadIfNeeded(160, 160, value=0, border_mode=1),
                                        A.CenterCrop(160, 160),
                                        A.HorizontalFlip(),
                                        A.Rotate(limit=5),
                                        ])

            transform_color = A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=.15, contrast_limit=.15),
                    A.GaussianBlur(blur_limit=7),
                    A.GaussNoise(var_limit=.001, )
            ])
        else:
            transform_shape = transform_color = None

        super(brain_dataset,self).__init__(data_dir, sample = True, transform_shape=transform_shape,
                                                 transform_color = transform_color, **kwargs)



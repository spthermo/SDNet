import os
import sys

import nibabel as nib
import numpy as np
import cv2

import utils.data_utils
from loaders.base_loader import Loader
from skimage import transform
sys.path.append('..')
sys.path.append('./loaders')
from loaders.data import Data
from utils import data_utils
#from parameters import conf
import logging


class ACDCLoader(Loader):
    def __init__(self, data_path):
        super(ACDCLoader, self).__init__()
        self.num_masks = 3
        self.num_volumes = 100
        self.input_shape = (224, 224, 1)
        self.data_folder = data_path
        self.log = logging.getLogger('acdc')

    def splits(self):
        '''
        :return: an array of splits into validation, test and train indices
        '''

        splits = [
            {'validation': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             'test': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
             'training': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                          47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                          64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                          81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
             },
            {'validation': [85, 13, 9, 74, 73, 68, 59, 79, 47, 80, 14, 95, 25, 92, 87],
             'test': [54, 55, 99, 63, 91, 24, 51, 3, 64, 43, 61, 66, 96, 27, 76],
             'training': [46, 57, 49, 34, 17, 8, 19, 28, 97, 1, 90, 22, 88, 45, 12, 4, 5,
                          75, 53, 94, 62, 86, 35, 58, 82, 37, 84, 93, 6, 33, 15, 81, 23, 48,
                          71, 70, 11, 77, 36, 60, 31, 65, 32, 78, 98, 52, 100, 42, 38, 2, 20,
                          69, 26, 18, 40, 50, 16, 7, 41, 10, 83, 21, 39, 72, 56, 67, 44, 30, 89, 29]
             },
            {'validation': [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
             'test': [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
             'training': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                          100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 62, 63,
                          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 79, 80,
                          81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
             },
            {'validation': [20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 37, 33, 34, 35, 36],
             'test': [38, 39, 40, 41, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55],
             'training': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                          21, 22, 32, 42, 44, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                          71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                          91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
             },
            {'validation': [11, 12, 13, 14, 15],
             'test': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
             'training': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                          47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                          64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                          81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
             }
        ]

        return splits

    def load_scanner_type(self):
        """
        Load a text file of volume to MR scanner type (1.5T or 3T), for example:
         1, 1.5
         2, 1.5
         3, 3
         4, 3
         etc
        :return: a map of volume to scanner type
        """
        folder_name = self.data_folder + '/acdc_t1_5_vs_T3.txt'
        if not os.path.exists(folder_name):
            self.log.warn('Scanner type text does not exist. Initialising with -1')
            vol_scanner = {v: -1 for v in self.volumes}
        else:
            vol_scanner = {}
            with open(folder_name, 'r') as f:
                for l in f:
                    pat, sc = l.split('\t')
                    vol_scanner[int(pat[-3:])] = float(sc.replace('\n', '').replace(',', '.'))

        return vol_scanner

    def load_labelled_data(self, split, split_type, modality='MR', normalise=False, value_crop=True, downsample=1):
        if split < 0 or split > 4:
            raise ValueError('Invalid value for split: %d. Allowed values are 0, 1, 2.' % split)
        if split_type not in ['training', 'validation', 'test', 'all']:
            raise ValueError('Invalid value for split_type: %s. Allowed values are training, validation, test, all'
                             % split_type)

        npz_prefix = 'norm_' if normalise else 'unnorm_'

        # If numpy arrays are not saved, load and process raw data
        if not os.path.exists(os.path.join(self.data_folder, npz_prefix + 'acdc_images.npz')):
            images, masks_lv, masks_rv, masks_myo, index = self.load_raw_labelled_data(normalise, value_crop)

            # save numpy arrays
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'acdc_images'), images)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'acdc_masks_lv'), masks_lv)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'acdc_masks_rv'), masks_rv)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'acdc_masks_myo'), masks_myo)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'acdc_index'), index)
        # Load data from saved numpy arrays
        else:
            images    = np.load(os.path.join(self.data_folder, npz_prefix + 'acdc_images.npz'))['arr_0']
            masks_lv  = np.load(os.path.join(self.data_folder, npz_prefix + 'acdc_masks_lv.npz'))['arr_0']
            masks_rv  = np.load(os.path.join(self.data_folder, npz_prefix + 'acdc_masks_rv.npz'))['arr_0']
            masks_myo = np.load(os.path.join(self.data_folder, npz_prefix + 'acdc_masks_myo.npz'))['arr_0']
            index     = np.load(os.path.join(self.data_folder, npz_prefix + 'acdc_index.npz'))['arr_0']

        assert images is not None and masks_myo is not None and masks_lv is not None and masks_rv is not None \
               and index is not None, 'Could not find saved data'

        assert images.max() == 1 and images.min() == -1, \
            'Images max=%.3f, min=%.3f' % (images.max(), images.min())

        self.log.debug('Loaded compressed acdc data of shape: ' + str(images.shape) + ' ' + str(index.shape))

        vol_scanner = self.load_scanner_type()

        # Case to load data from all splits.
        if split_type == 'all':
            masks = np.concatenate([masks_myo, masks_lv, masks_rv], axis=-1)
            scanner = index.copy()
            for v in self.volumes:
                scanner[index == v] = vol_scanner[v]
            return Data(images, masks, index, scanner, downsample)

        # Select images belonging to the volumes of the split_type (training, validation, test)
        volumes = self.splits()[split][split_type]
        
        images = np.concatenate([images[index == v] for v in volumes])
        masks = np.concatenate([masks_myo, masks_lv, masks_rv], axis=-3)
        assert masks.max() == 1 and masks.min() == 0, 'Masks max=%.3f, min=%.3f' % (masks.max(), masks.min())
        masks = np.concatenate([masks[index == v] for v in volumes])
        assert images.shape[0] == masks.shape[0]
        
        # create a volume index
        index = np.concatenate([index[index == v] for v in volumes])
        assert images.shape[0] == index.shape[0]
        
        scanner = index.copy()
        for v in volumes:
            scanner[index == v] = vol_scanner[v]

        self.log.debug(split_type + ' set: ' + str(images.shape))

        # spthermo insertion for resizing acdc data
        # images_resized = np.zeros((images.shape[0], 1, 64, 64))
        # masks_resized = np.zeros((masks.shape[0], 3, 64, 64))
        # for idx in range(images.shape[0]):
        #     resized = cv2.resize(images[idx].squeeze(), dsize=(64, 64), interpolation=cv2.INTER_AREA)
        #     resized = -1 + (resized - np.min(resized)*2)/(np.max(resized) - np.min(resized))
        #     images_resized[idx] = np.expand_dims(resized, axis=0)

        #     m_resized = cv2.resize(masks[idx][0], dsize=(64, 64), interpolation=cv2.INTER_AREA)
        #     m_resized = -1 + (m_resized - np.min(m_resized)*2)/(np.max(m_resized) - np.min(m_resized))
        #     masks_resized[idx][0] = np.expand_dims(m_resized, axis=0)
        #     masks_resized[idx][1] = np.expand_dims(m_resized, axis=0)
        #     masks_resized[idx][2] = np.expand_dims(m_resized, axis=0)

        return Data(images, masks, index, scanner, downsample)

    def load_unlabelled_data(self, split, split_type, modality='MR', normalise=True, value_crop=True):
        images, index = self.base_load_unlabelled_images('acdc', split, split_type, False, normalise, value_crop)
        masks = np.zeros(shape=(images.shape[:-1]) + (1,))
        vol_scanner = self.load_scanner_type()
        scanner = index.copy()
        for v in self.volumes:
            scanner[index == v] = vol_scanner[v]
        return Data(images, masks, index, scanner)

    def load_all_data(self, split, split_type, modality='MR', normalise=True, value_crop=True):
        images, index = self.base_load_unlabelled_images('acdc', split, split_type, True, normalise, value_crop)
        masks = np.zeros(shape=(images.shape[:-1]) + (1,))

        vol_scanner = self.load_scanner_type()
        scanner = index.copy()
        for v in self.volumes:
            scanner[index == v] = vol_scanner[v]
        return Data(images, masks, index, scanner)

    def load_raw_labelled_data(self, normalise=True, value_crop=True):
        self.log.debug('Loading acdc data from original location')
        images, masks_lv, masks_rv, masks_myo, index = [], [], [], [], []

        for patient_i in self.volumes:
            patient = 'patient%03d' % patient_i
            patient_folder = os.path.join(self.data_folder, patient)

            gt = [f for f in os.listdir(patient_folder) if 'gt' in f and f.startswith(patient + '_frame')]
            ims = [f.replace('_gt', '') for f in gt]

            for i in range(len(ims)):
                im = self.process_raw_image(ims[i], patient_folder, value_crop, normalise)
                im = np.expand_dims(im, axis=-1)

                m = self.resample_raw_image(gt[i], patient_folder, binary=True)
                m = np.expand_dims(m, axis=-1)

                images.append(im)

                # convert 3-dim mask array to 3 binary mask arrays for lv, rv, myo
                m_lv = m.copy()
                m_lv[m != 3] = 0
                m_lv[m == 3] = 1
                masks_lv.append(m_lv)

                m_rv = m.copy()
                m_rv[m != 1] = 0
                m_rv[m == 1] = 1
                masks_rv.append(m_rv)

                m_myo = m.copy()
                m_myo[m != 2] = 0
                m_myo[m == 2] = 1
                masks_myo.append(m_myo)

                index += [patient_i] * im.shape[2]

        assert len(images) == len(masks_myo)

        # move slice axis to the first position
        images    = [np.moveaxis(im, 2, 0) for im in images]
        masks_lv  = [np.moveaxis(m, 2, 0) for m in masks_lv]
        masks_rv  = [np.moveaxis(m, 2, 0) for m in masks_rv]
        masks_myo = [np.moveaxis(m, 2, 0) for m in masks_myo]

        # crop images and masks to the same pixel dimensions and concatenate all data
        images_cropped, masks_lv_cropped = utils.data_utils.crop_same(images, masks_lv, (224, 224))
        _, masks_rv_cropped = utils.data_utils.crop_same(images, masks_rv, (224, 224))
        _, masks_myo_cropped = utils.data_utils.crop_same(images, masks_myo, (224, 224))

        for i in range (len(images_cropped)):
            images_cropped[i] = utils.data_utils.normalise(images_cropped[i], -1, 1)

        images_cropped    = np.concatenate(images_cropped, axis=0)
        masks_lv_cropped  = np.concatenate(masks_lv_cropped, axis=0)
        masks_rv_cropped  = np.concatenate(masks_rv_cropped, axis=0)
        masks_myo_cropped = np.concatenate(masks_myo_cropped, axis=0)
        index             = np.array(index)

        self.log.debug(str(images[0].shape) + ', ' + str(masks_lv[0].shape))

        return images_cropped.transpose(0,3,1,2), masks_lv_cropped.transpose(0,3,1,2), masks_rv_cropped.transpose(0,3,1,2), masks_myo_cropped.transpose(0,3,1,2), index

    def resample_raw_image(self, mask_fname, patient_folder, binary=True):
        """
        Load raw data (image/mask) and resample to fixed resolution.
        :param mask_fname:     filename of mask
        :param patient_folder: folder containing patient data
        :param binary:         boolean to define binary masks or not
        :return: the resampled image
        """
        m_nii_fname = os.path.join(patient_folder, mask_fname)
        new_res = (1.37, 1.37)
        print('Resampling %s at resolution %s to file %s' % (m_nii_fname, str(new_res), new_res))
        im_nii = nib.load(m_nii_fname)
        im_data = im_nii.get_data()
        voxel_size = im_nii.header.get_zooms()

        scale_vector = [voxel_size[i] / new_res[i] for i in range(len(new_res))]
        order = 0 if binary else 1

        result = []
        for i in range(im_data.shape[-1]):
            im = im_data[..., i]
            rescaled = transform.rescale(im, scale_vector, order=order, preserve_range=True, mode='constant')
            result.append(np.expand_dims(rescaled, axis=-1))
        return np.concatenate(result, axis=-1)


    def process_raw_image(self, im_fname, patient_folder, value_crop, normalise):
        """
        Normalise and crop extreme values of an image
        :param im_fname:        filename of the image
        :param patient_folder:  folder of patient data
        :param value_crop:      True/False to crop values between 5/95 percentiles
        :param normalise:       True/False normalise images
        :return:                a processed image
        """
        im = self.resample_raw_image(im_fname, patient_folder, binary=False)

        # crop to 5-95 percentile
        if value_crop:
            p5 = np.percentile(im.flatten(), 5)
            p95 = np.percentile(im.flatten(), 95)
            im = np.clip(im, p5, p95)

        # normalise to -1, 1
        if normalise:
            im = utils.data_utils.normalise(im, -1, 1)

        return im

    def load_raw_unlabelled_data(self, include_labelled=True, normalise=True, value_crop=True):
        self.log.debug('Loading unlabelled acdc data from original location')
        images, index = [], []

        for patient_i in self.volumes:
            patient = 'patient%03d' % patient_i
            self.log.debug('Loading patient %s' % patient)
            patient_folder = os.path.join(self.data_folder, patient)

            im_name = patient + '_4d.nii.gz'
            im = self.process_raw_image(im_name, patient_folder, value_crop, normalise)

            frames = range(im.shape[-1])
            if not include_labelled:
                gt = [f for f in os.listdir(patient_folder) if 'gt' in f and not f.startswith('._')]
                gt_ims = [f.replace('_gt', '') for f in gt if not f.startswith('._')]

                exclude_frames = [int(gt_im.split('.')[0].split('frame')[1]) for gt_im in gt_ims]
                frames = [f for f in range(im.shape[-1]) if f not in exclude_frames]

            for frame in frames:
                im_res = im[:, :, :, frame]
                if im_res.sum() == 0:
                    print('Skipping blank images')
                    continue
                images.append(im_res)
                index += [patient_i] * im_res.shape[-1]

        images = [np.expand_dims(np.moveaxis(im, 2, 0), axis=3) for im in images]
        zeros = [np.zeros(im.shape) for im in images]
        images_cropped, _ = utils.data_utils.crop_same(images, zeros, (224, 224))
        images_cropped = np.concatenate(images_cropped, axis=0)[..., 0]
        index = np.array(index)

        return images_cropped.transpose(0,3,1,2), index


# if __name__ == '__main__':
#     loader = ACDCLoader()
#     data = loader.load_labelled_data(0, 'test')
#     print(data.images.shape, data.masks.shape)
#     print(data.volumes(), len(data.volumes()))

    # data = loader.load_unlabelled_data(0, 'all')
    # print(data.images.shape)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(data.masks[30, :, :, 0] + data.masks[30, :, :, 1] * 2 + data.masks[30, :, :, 2] * 3)
    # plt.show()

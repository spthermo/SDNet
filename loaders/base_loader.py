import torch
import numpy as np
import os

class Loader(object):
    """
    Abstract class defining the behaviour of loaders for different datasets.
    """
    def __init__(self):
        self.num_masks   = 0
        self.num_volumes = 0
        self.input_shape = (None, None, 1)
        self.data_folder = None
        self.volumes = sorted(self.splits()[0]['training'] +
                              self.splits()[0]['validation'] +
                              self.splits()[0]['test'])
        self.log = None

    def base_load_unlabelled_images(self, dataset, split, split_type, include_labelled, normalise, value_crop):
        npz_prefix_type = 'ul_' if not include_labelled else 'all_'
        npz_prefix = npz_prefix_type + 'norm_' if normalise else npz_prefix_type + 'unnorm_'

        # Load saved numpy array
        if os.path.exists(os.path.join(self.data_folder, npz_prefix + dataset + '_images.npz')):
            images = np.load(os.path.join(self.data_folder, npz_prefix + dataset + '_images.npz'))['arr_0']
            index  = np.load(os.path.join(self.data_folder, npz_prefix + dataset + '_index.npz'))['arr_0']
            self.log.debug('Loaded compressed ' + dataset + ' unlabelled data of shape ' + str(images.shape))
        # Load from source
        else:
            images, index = self.load_raw_unlabelled_data(include_labelled, normalise, value_crop)
            images = np.expand_dims(images, axis=3)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + dataset + '_images'), images)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + dataset + '_index'), index)

        assert split_type in ['training', 'validation', 'test', 'all'], 'Unknown split_type: ' + split_type

        if split_type == 'all':
            return images, index

        volumes = self.splits()[split][split_type]
        images = np.concatenate([images[index == v] for v in volumes])
        index  = np.concatenate([index[index==v] for v in volumes])
        return images, index

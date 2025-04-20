import os
import warnings
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from datetime import datetime
to_date   = lambda string: datetime.strptime(string, '%Y-%m-%d')
S1_LAUNCH = to_date('2014-04-03')

import torch
import rasterio
from torch.utils.data import Dataset

def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif

def read_img(tif):
    return tif.read().astype(np.float32)

def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img      = (img - oldMin) / oldRange
    return img

def process_MS(img):                                   # 13 channel!!!!!!!!!!!!
    intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
    img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
    img = rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    img = np.nan_to_num(img)
    return img[[3,2,1], :, :]

def process_SAR(img):
    dB_min, dB_max = [-25.0, -32.5], [0, 0]
    img = np.concatenate([((np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                        ((np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    img = np.nan_to_num(img)
    return img


class SEN12MSCR(Dataset):
    def __init__(self, root, split="all", region='all', sample_type='pretrain'):

        self.root_dir = root                                # set root directory which contains all ROI
        self.region   = region                              # region according to which the ROI are selected 
        if self.region != 'all': raise NotImplementedError  # currently only supporting 'all'
        self.ROI      = {'ROIs1158': ['106'],
                         'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
                         'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
                         'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146']}

        self.splits         = {}
        self.splits['train']= ['ROIs1970_fall_s1/s1_3', 'ROIs1970_fall_s1/s1_22', 'ROIs1970_fall_s1/s1_148', 'ROIs1970_fall_s1/s1_107', 'ROIs1970_fall_s1/s1_1', 'ROIs1970_fall_s1/s1_114', 
                               'ROIs1970_fall_s1/s1_135', 'ROIs1970_fall_s1/s1_40', 'ROIs1970_fall_s1/s1_42', 'ROIs1970_fall_s1/s1_31', 'ROIs1970_fall_s1/s1_149', 'ROIs1970_fall_s1/s1_64', 
                               'ROIs1970_fall_s1/s1_28', 'ROIs1970_fall_s1/s1_144', 'ROIs1970_fall_s1/s1_57', 'ROIs1970_fall_s1/s1_35', 'ROIs1970_fall_s1/s1_133', 'ROIs1970_fall_s1/s1_30', 
                               'ROIs1970_fall_s1/s1_134', 'ROIs1970_fall_s1/s1_141', 'ROIs1970_fall_s1/s1_112', 'ROIs1970_fall_s1/s1_116', 'ROIs1970_fall_s1/s1_37', 'ROIs1970_fall_s1/s1_26', 
                               'ROIs1970_fall_s1/s1_77', 'ROIs1970_fall_s1/s1_100', 'ROIs1970_fall_s1/s1_83', 'ROIs1970_fall_s1/s1_71', 'ROIs1970_fall_s1/s1_93', 'ROIs1970_fall_s1/s1_119', 
                               'ROIs1970_fall_s1/s1_104', 'ROIs1970_fall_s1/s1_136', 'ROIs1970_fall_s1/s1_6', 'ROIs1970_fall_s1/s1_41', 'ROIs1970_fall_s1/s1_125', 'ROIs1970_fall_s1/s1_91', 
                               'ROIs1970_fall_s1/s1_131', 'ROIs1970_fall_s1/s1_120', 'ROIs1970_fall_s1/s1_110', 'ROIs1970_fall_s1/s1_19', 'ROIs1970_fall_s1/s1_14', 'ROIs1970_fall_s1/s1_81', 
                               'ROIs1970_fall_s1/s1_39', 'ROIs1970_fall_s1/s1_109', 'ROIs1970_fall_s1/s1_33', 'ROIs1970_fall_s1/s1_88', 'ROIs1970_fall_s1/s1_11', 'ROIs1970_fall_s1/s1_128', 
                               'ROIs1970_fall_s1/s1_142', 'ROIs1970_fall_s1/s1_122', 'ROIs1970_fall_s1/s1_4', 'ROIs1970_fall_s1/s1_27', 'ROIs1970_fall_s1/s1_147', 'ROIs1970_fall_s1/s1_85', 
                               'ROIs1970_fall_s1/s1_82', 'ROIs1970_fall_s1/s1_105', 'ROIs1158_spring_s1/s1_9', 'ROIs1158_spring_s1/s1_1', 'ROIs1158_spring_s1/s1_124', 'ROIs1158_spring_s1/s1_40', 
                               'ROIs1158_spring_s1/s1_101', 'ROIs1158_spring_s1/s1_21', 'ROIs1158_spring_s1/s1_134', 'ROIs1158_spring_s1/s1_145', 'ROIs1158_spring_s1/s1_141', 'ROIs1158_spring_s1/s1_66', 
                               'ROIs1158_spring_s1/s1_8', 'ROIs1158_spring_s1/s1_26', 'ROIs1158_spring_s1/s1_77', 'ROIs1158_spring_s1/s1_113', 'ROIs1158_spring_s1/s1_100', 
                               'ROIs1158_spring_s1/s1_117', 'ROIs1158_spring_s1/s1_119', 'ROIs1158_spring_s1/s1_6', 'ROIs1158_spring_s1/s1_58', 'ROIs1158_spring_s1/s1_120', 'ROIs1158_spring_s1/s1_110', 
                               'ROIs1158_spring_s1/s1_126', 'ROIs1158_spring_s1/s1_115', 'ROIs1158_spring_s1/s1_121', 'ROIs1158_spring_s1/s1_39', 'ROIs1158_spring_s1/s1_109', 'ROIs1158_spring_s1/s1_63', 
                               'ROIs1158_spring_s1/s1_75', 'ROIs1158_spring_s1/s1_132', 'ROIs1158_spring_s1/s1_128', 'ROIs1158_spring_s1/s1_142', 'ROIs1158_spring_s1/s1_15', 'ROIs1158_spring_s1/s1_45', 
                               'ROIs1158_spring_s1/s1_97', 'ROIs1158_spring_s1/s1_147', 'ROIs1868_summer_s1/s1_90', 'ROIs1868_summer_s1/s1_87', 'ROIs1868_summer_s1/s1_25', 'ROIs1868_summer_s1/s1_124', 
                               'ROIs1868_summer_s1/s1_114', 'ROIs1868_summer_s1/s1_135', 'ROIs1868_summer_s1/s1_40', 'ROIs1868_summer_s1/s1_101', 'ROIs1868_summer_s1/s1_42', 
                               'ROIs1868_summer_s1/s1_31', 'ROIs1868_summer_s1/s1_36', 'ROIs1868_summer_s1/s1_139', 'ROIs1868_summer_s1/s1_56', 'ROIs1868_summer_s1/s1_133', 'ROIs1868_summer_s1/s1_55', 
                               'ROIs1868_summer_s1/s1_43', 'ROIs1868_summer_s1/s1_113', 'ROIs1868_summer_s1/s1_76', 'ROIs1868_summer_s1/s1_123', 'ROIs1868_summer_s1/s1_143', 
                               'ROIs1868_summer_s1/s1_93', 'ROIs1868_summer_s1/s1_125', 'ROIs1868_summer_s1/s1_89', 'ROIs1868_summer_s1/s1_120', 'ROIs1868_summer_s1/s1_126', 'ROIs1868_summer_s1/s1_72', 
                               'ROIs1868_summer_s1/s1_115', 'ROIs1868_summer_s1/s1_121', 'ROIs1868_summer_s1/s1_146', 'ROIs1868_summer_s1/s1_140', 'ROIs1868_summer_s1/s1_95', 
                               'ROIs1868_summer_s1/s1_102', 'ROIs1868_summer_s1/s1_7', 'ROIs1868_summer_s1/s1_11', 'ROIs1868_summer_s1/s1_132', 'ROIs1868_summer_s1/s1_15', 'ROIs1868_summer_s1/s1_137', 
                               'ROIs1868_summer_s1/s1_4', 'ROIs1868_summer_s1/s1_27', 'ROIs1868_summer_s1/s1_147', 'ROIs1868_summer_s1/s1_86', 'ROIs1868_summer_s1/s1_47', 'ROIs2017_winter_s1/s1_68', 
                               'ROIs2017_winter_s1/s1_25', 'ROIs2017_winter_s1/s1_62', 'ROIs2017_winter_s1/s1_135', 'ROIs2017_winter_s1/s1_42', 'ROIs2017_winter_s1/s1_64', 'ROIs2017_winter_s1/s1_21', 
                               'ROIs2017_winter_s1/s1_55', 'ROIs2017_winter_s1/s1_112', 'ROIs2017_winter_s1/s1_116', 'ROIs2017_winter_s1/s1_8', 'ROIs2017_winter_s1/s1_59', 'ROIs2017_winter_s1/s1_49', 
                               'ROIs2017_winter_s1/s1_104',  'ROIs2017_winter_s1/s1_81', 'ROIs2017_winter_s1/s1_146', 'ROIs2017_winter_s1/s1_75', 
                               'ROIs2017_winter_s1/s1_94', 'ROIs2017_winter_s1/s1_102', 'ROIs2017_winter_s1/s1_61', 'ROIs2017_winter_s1/s1_47',
                               'ROIs1868_summer_s1/s1_100', # note: this ROI is also used for testing in SEN12MS-CR-TS. If you wish to combine both datasets, please comment out this line
                               ]
        self.splits['val']  = ['ROIs2017_winter_s1/s1_22', 'ROIs1868_summer_s1/s1_19', 'ROIs1970_fall_s1/s1_65', 'ROIs1158_spring_s1/s1_17', 'ROIs2017_winter_s1/s1_107', 
                               'ROIs1868_summer_s1/s1_80', 'ROIs1868_summer_s1/s1_127', 'ROIs2017_winter_s1/s1_130', 'ROIs1868_summer_s1/s1_17', 'ROIs2017_winter_s1/s1_84'] 
        self.splits['test'] = ['ROIs1158_spring_s1/s1_106', 'ROIs1158_spring_s1/s1_123', 'ROIs1158_spring_s1/s1_140', 'ROIs1158_spring_s1/s1_31', 'ROIs1158_spring_s1/s1_44', 
                               'ROIs1868_summer_s1/s1_119', 'ROIs1868_summer_s1/s1_73', 'ROIs1970_fall_s1/s1_139', 'ROIs2017_winter_s1/s1_108', 'ROIs2017_winter_s1/s1_63']

        self.splits["all"]  = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.split = split
        
        assert split in ['all', 'train', 'val', 'test'], "Input dataset must be either assigned as all, train, test, or val!"
        assert sample_type in ['pretrain'], "Input data must be pretrain!"
        
        self.modalities     = ["S1", "S2"]
        self.sample_type    = sample_type   # e.g. 'pretrain'

        self.time_points    = range(1)
        self.n_input_t      = 1             # specifies the number of samples, if only part of the time series is used as an input

        self.paths          = self.get_paths()
        self.n_samples      = len(self.paths)

        # raise a warning if no data has been found
        if not self.n_samples: self.throw_warn()

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region {self.region}')

        paths = []
        seeds_S1 = natsorted([s1dir for s1dir in os.listdir(self.root_dir) if "_s1" in s1dir])
        for seed in tqdm(seeds_S1):
            rois_S1 = natsorted(os.listdir(os.path.join(self.root_dir, seed)))
            for roi in rois_S1:
                roi_dir  = os.path.join(self.root_dir, seed, roi)
                paths_S1        = natsorted([os.path.join(roi_dir, s1patch) for s1patch in os.listdir(roi_dir)])
                paths_S2        = [patch.replace('/s1', '/s2').replace('_s1', '_s2') for patch in paths_S1]
                paths_S2_cloudy = [patch.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy') for patch in paths_S1]

                for pdx, _ in enumerate(paths_S1):
                    # omit patches that are potentially unpaired
                    if not all([os.path.isfile(paths_S1[pdx]), os.path.isfile(paths_S2[pdx]), os.path.isfile(paths_S2_cloudy[pdx])]): continue
                    # don't add patch if not belonging to the selected split
                    if not any([split_roi in paths_S1[pdx] for split_roi in self.splits[self.split]]): continue
                    sample = {"S1":         paths_S1[pdx],
                              "S2":         paths_S2[pdx],
                              "S2_cloudy":  paths_S2_cloudy[pdx]}
                    paths.append(sample)
        return paths

    def __getitem__(self, pdx):  # get the triplet of patch with ID pdx
        s1_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S1']))
        s2_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S2']))
        s2_cloudy_tif   = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S2_cloudy']))
        s1              = torch.from_numpy(process_SAR(read_img(s1_tif))).float()
        s2              = torch.from_numpy(process_MS(read_img(s2_tif))).float()           # note: pre-processing happens after cloud detection
        s2_cloudy       = torch.from_numpy(process_MS(read_img(s2_cloudy_tif))).float()    # note: pre-processing happens after cloud detection

        sample = {'input': {'S1': s1,
                            'S2': s2_cloudy,
                            },
                    'target': {'S2': s2}
                    }
        return sample

    def throw_warn(self):
        warnings.warn("""No data samples found! Please use the following directory structure:

        path/to/your/SEN12MSCR/directory:
            ├───ROIs1158_spring_s1
            |   ├─s1_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s1_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2
            |   ├─s2_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2_cloudy
            |   ├─s2_cloudy_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_cloudy_1_p407.tif
            |   |   |...
            |    ...
            ...

        Note: Please arrange the dataset in a format as e.g. provided by the script dl_data.sh.
        """)

    def __len__(self):
        # length of generated list
        return self.n_samples
import cv2
import os
import ast
import tqdm
import math
import random
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from skimage.io import imread
from multiprocessing import Pool

from utils.bbox_tools import bb_tuple2dict_2d,bbox_inside_2d,iou_2d

cv2.setNumThreads(0)

class OiDataset(data.Dataset):
    def __init__(self,
                 mode = 'train', # 'train' or val'
                 random_state = 42,
                 fold = 0,
                 size_ratio = 1.0,
                 
                 mean = (0.485, 0.456, 0.406),
                 std = (0.229, 0.224, 0.225),
                 
                 weight_log_base = 2,
                 min_class_weight = 1,
                 img_size_cluster = 0,
                 
                 fixed_size = (224,224),
                 
                 data_folder = '../../../hdd/open_images/',
                 # train_imgs_folder = '../../../hdd/open_images/train/',
                 # val_imgs_folder = '../../../hdd/open_images/train/',
                 # train_imgs_folder = '../data/train/train/',
                 train_imgs_folder = '../data/train/train/',
                 
                 
                 val_imgs_folder = '../data/train/train/',                 
                 
                 label_list_path = '../data/label_list',
                 label_counts_path = '../data/label_counts.csv',
                 e2e_resize_dict_path = '../data/e2e_resize_dict.pickle',
                 imgid_size_dict_path = '../data/imgid_size.pickle',
                 # multi_label_dataset_path = '../data/multi_label_imgs_area_classes.csv',
                 multi_label_dataset_path = '../data/multi_label_imgs_class_count_corrected_relatons_ohe.csv',
                  
                 return_img_id = False,
                 stratify_label = 'class_count',
                 prob = 0.25,
                 oversampling_floor = 8,
                 oversampling = False,
                 
                multi_class = True
                ):
        
        self.fold = fold
        self.mode = mode
        self.mean = mean
        self.std = std
        self.size_ratio = size_ratio
        self.random_state = random_state
        self.weight_log_base = weight_log_base
        self.min_class_weight = min_class_weight
        
        self.return_img_id = return_img_id
        
        self.fixed_size = fixed_size
        
        self.prob = prob
        self.oversampling_floor = oversampling_floor
        
        cluster_dict = {
            0:'(512, 1024)',
            1:'(1024, 512)',
            2:'(768, 768)'
        }
        
        self.data_folder = data_folder
        self.train_imgs_folder = train_imgs_folder
        self.val_imgs_folder = val_imgs_folder
        
        with open(label_list_path, 'rb') as handle:
            self.label_list = pickle.load(handle)

        with open(e2e_resize_dict_path, 'rb') as handle:
            self.e2e_resize_dict = pickle.load(handle)
            
        with open(imgid_size_dict_path, 'rb') as handle:
            self.imgid_size_dict = pickle.load(handle)
            
        self.label_counts = pd.read_csv(label_counts_path, names=['class','count'])
        multi_label_dataset = pd.read_csv(multi_label_dataset_path)

        if img_size_cluster != 'sample':
            # choose only images of one size
            multi_label_dataset = multi_label_dataset[multi_label_dataset.target_resl == cluster_dict[img_size_cluster]]
       
        self.ohe_values = list(multi_label_dataset.ohe_vectors.values)
        self.stratify_values = list((multi_label_dataset[stratify_label]).astype('int').values)
        self.img_ids = list(multi_label_dataset['img_id'].values)
        
        skf = StratifiedKFold(n_splits=5,
                              shuffle = True,
                              random_state = self.random_state)
        
        f1, f2, f3, f4, f5 = skf.split(self.img_ids,
                                       self.stratify_values)
        
        folds = [f1, f2, f3, f4, f5]
        if self.mode == 'train':
            self.train_idx = list(folds[self.fold][0])
            train_idx_dict = dict(zip(self.train_idx,
                                      range(0,len(self.train_idx))))            
         
            if img_size_cluster == 'sample':
                # save indexes of each cluster
                # to be used later in sampling process
                # leave only the train indexes
                cluster_indices = []
                for cluster,img_size in cluster_dict.items():
                    # leave only the train indexes
                    condition = (multi_label_dataset.target_resl == img_size)&(multi_label_dataset.index.isin(self.train_idx))
                    cluster_list = list(multi_label_dataset[condition].index.values)
                    # reindex the cluster indices with respect to the train/val split values
                    cluster_list = [train_idx_dict[_]  for _ in cluster_list]
                    cluster_indices.append(cluster_list)                    

                self.cluster_indices = cluster_indices
        elif self.mode == 'val':
            self.val_idx = list(folds[self.fold][1])
            val_idx_dict = dict(zip(self.val_idx,
                                    range(0,len(self.val_idx))))
            
            if img_size_cluster == 'sample':
                # save indexes of each cluster
                # to be used later in sampling process
                # leave only the train indexes
                cluster_indices = []
                for cluster,img_size in cluster_dict.items():
                    # leave only the train indexes
                    condition = (multi_label_dataset.target_resl == img_size)&(multi_label_dataset.index.isin(self.val_idx))
                    cluster_list = list(multi_label_dataset[condition].index.values)
                    # reindex the cluster indices with respect to the train/val split values
                    cluster_list = [val_idx_dict[_]  for _ in cluster_list]
                    cluster_indices.append(cluster_list)
                    
                self.cluster_indices = cluster_indices            
        
        del multi_label_dataset
        self.produce_weights()
        
        if oversampling:
            # use this later with a sampler
            self.dataset_oversampling_list = self.produce_oversampling_weights()
            
            if self.mode == 'train':
                # first we need to revert the train/val indexing
                # then we need to pull the necessary oversampling values
                train_idx_dict_reverse = dict(zip(range(0,len(self.train_idx)),
                                                  self.train_idx)) 
                self.oversampling_indices = []
                for cluster in self.cluster_indices:
                    self.oversampling_indices.append([self.dataset_oversampling_list[train_idx_dict_reverse[_]] for _ in cluster])
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)
    def produce_weights(self):
        max_log = self.label_counts['count'].apply(lambda x: math.log(x, self.weight_log_base)).max()
        self.label_counts['class_weight'] = self.label_counts['count'].apply(lambda x: self.min_class_weight+(max_log-math.log(x, self.weight_log_base)))

        label_weight_dict = dict(zip(self.label_counts['class'].values,self.label_counts['class_weight'].values))
        self.label_weight_list = np.asarray([label_weight_dict[_] for _ in self.label_list])
    def produce_oversampling_weights(self):
        # classes will be oversampled only if their log count is lower than specified
        oversampling_list = [int(math.ceil(max(_-self.oversampling_floor,1))**1.5) for _ in self.label_weight_list]
        
        print('Calculating oversampling weights - it is slow due to ast literal eval')
        with Pool(6) as p:
            ohe_lists = list(tqdm.tqdm(p.imap(leval, self.ohe_values), total=len(self.ohe_values)))

        def produce_oversampling_factor(ohe,oversampling_list):
            factors = []
            for i,_ in enumerate(ohe):
                if _== 1:
                    factors.append(oversampling_list[i])
            return max(factors)              
            
        dataset_oversampling_list = [produce_oversampling_factor(_, oversampling_list) for _ in ohe_lists]
        return dataset_oversampling_list
    def __getitem__(self, idx):
        if self.mode == 'train':
            img_id = self.img_ids[self.train_idx[idx]]
            img_path = os.path.join(self.train_imgs_folder,img_id)+'.jpg'
            ohe_values = self.ohe_values[self.train_idx[idx]]
        elif self.mode == 'val':
            img_id = self.img_ids[self.val_idx[idx]]
            img_path = os.path.join(self.val_imgs_folder,img_id)+'.jpg'
            ohe_values = self.ohe_values[self.val_idx[idx]]
        
        ohe_values = np.asarray(ast.literal_eval(ohe_values))
        target_size = self.e2e_resize_dict[self.imgid_size_dict[img_id]]
        img = self.preprocess_img(img_path,target_size)
        
        if img is None:
            # do not return anything
            pass
        else:
            # add failsafe values here
            
            if self.return_img_id == False:
                return_tuple = (img,
                                ohe_values,
                                self.label_weight_list)                
            else:
                return_tuple = (img,
                                ohe_values,
                                self.label_weight_list,
                                img_id)
            return return_tuple 
    def preprocess_img(self,
                       img_path,
                       target_size,
                       ):

        final_size =  [int(_ * self.size_ratio) for _ in target_size]
        img = imread(img_path)

        # gray-scale img
        if len(img.shape)==2:
            # convert grayscale images to RGB
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # gif image
        elif len(img.shape)==1:
            img = img[0]
        # alpha channel image
        elif img.shape[2] == 4:
            img = img[:,:,0:3]

        img = Image.fromarray(img)
        
        if self.preprocessing_type == 0:
            # fixed resize classic Imagenet Preprocessing
            preprocessing = transforms.Compose([
                            transforms.Resize(self.fixed_size),                
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ]) 
        elif self.preprocessing_type == 1:
            # a bit smarter Imagenet preprocessing
            # at first resize by a smaller size, then do a center crop
            preprocessing = transforms.Compose([
                            transforms.Resize(self.fixed_size[0]),
                            transforms.CenterCrop(self.fixed_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])               
        elif self.preprocessing_type == 2:
            # fixed resize to a cluster-defined size
            preprocessing = transforms.Compose([
                            transforms.Resize(final_size),                
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])          
        elif self.preprocessing_type == 3:
            # some additional augmentations
            add_transforms = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                              RandomResizedCropRect(final_size,scale=(0.8, 1.0), ratio=(0.8, 1.2), interpolation=2),
                              ]

            preprocessing = transforms.Compose([
                            transforms.Resize(final_size),

                            transforms.RandomApply(add_transforms, p=self.prob),
                            transforms.RandomHorizontalFlip(p=self.prob),
                            transforms.RandomVerticalFlip(p=self.prob),
                            transforms.RandomGrayscale(p=self.prob),

                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])
        else:
            raise ValueError('This augmentation is not supported')

class ImnetDataset(data.Dataset):
    def __init__(self,
                 mode = 'train', # 'train' or val'
                 random_state = 42,
                 fold = 0,
                 
                 size_ratio = 1.0,
                 preprocessing_type = 0, # 0,1,2
                 fixed_size = (224,224),
                 prob = 0.2,
                 
                 mean = (0.485, 0.456, 0.406),
                 std = (0.229, 0.224, 0.225),
                 
                 imgs_folder = '../../imagenet/',
                 df_path = '../data/imnet_cluster_df_short.feather',
                 return_img_id = False,
                 multi_class = False
                ):
        
        self.std = std
        self.fold = fold
        self.mode = mode
        self.mean = mean        

        self.size_ratio = size_ratio
        self.fixed_size = fixed_size
        self.random_state = random_state
        
        self.prob = prob
        self.return_img_id = return_img_id
        self.preprocessing_type = preprocessing_type
        
        self.multi_class = multi_class
        
        self.cluster_dict = {
            0: (384,512),
            1: (512, 512),
            2: (512, 384)
        }
        
        self.imgs_folder = imgs_folder
        imnet_df = pd.read_feather(df_path)
        self.label_list = sorted(imnet_df['class'].unique())

        self.label2name = dict(imnet_df[['class','label_name']].drop_duplicates().set_index('class')['label_name'])
        
        self.target_clusters = list(imnet_df['cluster'].values)
        self.stratify_values = list(imnet_df['cluster'].values)
        self.filenames = list(imnet_df['filename'].values)
        self.class_list = list(imnet_df['class'].values)
        
        skf = StratifiedKFold(n_splits=5,
                              shuffle = True,
                              random_state = self.random_state)
        
        f1, f2, f3, f4, f5 = skf.split(self.filenames,
                                       self.stratify_values)
        
        folds = [f1, f2, f3, f4, f5]
        
        if self.mode == 'train':
            self.train_idx = list(folds[self.fold][0])
            train_idx_dict = dict(zip(self.train_idx,
                                      range(0,len(self.train_idx))))            
         
            # save indexes of each cluster
            # to be used later in sampling process
            # leave only the train indexes
            cluster_indices = []
            for cluster,img_size in self.cluster_dict.items():
                # leave only the train indexes
                condition = (imnet_df.cluster == cluster)&(imnet_df.index.isin(self.train_idx))
                cluster_list = list(imnet_df[condition].index.values)
                # reindex the cluster indices with respect to the train/val split values
                cluster_list = [train_idx_dict[_]  for _ in cluster_list]
                cluster_indices.append(cluster_list)                    

            self.cluster_indices = cluster_indices
        elif self.mode == 'val':
            self.val_idx = list(folds[self.fold][1])
            val_idx_dict = dict(zip(self.val_idx,
                                    range(0,len(self.val_idx))))

            # save indexes of each cluster
            # to be used later in sampling process
            # leave only the train indexes
            cluster_indices = []
            for cluster,img_size in self.cluster_dict.items():
                # leave only the train indexes
                condition = (imnet_df.cluster == cluster)&(imnet_df.index.isin(self.val_idx))
                cluster_list = list(imnet_df[condition].index.values)
                # reindex the cluster indices with respect to the train/val split values
                cluster_list = [val_idx_dict[_]  for _ in cluster_list]
                cluster_indices.append(cluster_list)

            self.cluster_indices = cluster_indices            
        
        del imnet_df
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)
    def __getitem__(self, idx):
        if self.mode == 'train':
            img_index = self.train_idx[idx]
        elif self.mode == 'val':
            img_index = self.val_idx[idx]
        
        img_id = self.filenames[img_index]
        img_path = os.path.join(self.imgs_folder,img_id)
        class_name = self.class_list[img_index]
        
        if self.multi_class:
            ohe_values = np.zeros(len(self.label_list))
            # only one label for imagenet images
            ohe_values[self.label_list.index(class_name)] = 1
            target = ohe_values
        else:
            target = self.label_list.index(class_name)
        
        target_size = self.cluster_dict[self.target_clusters[img_index]]
        img = self.preprocess_img(img_path,target_size)
        
        if img is None:
            # do not return anything
            pass
        else:
            # add failsafe values here
            
            if self.return_img_id == False:
                return_tuple = (img,
                                target)                
            else:
                return_tuple = (img,
                                target,
                                img_id)
            return return_tuple 
    def preprocess_img(self,
                       img_path,
                       target_size,
                       ):

        final_size =  [int(_ * self.size_ratio) for _ in target_size]
        img = imread(img_path)

        # gray-scale img
        if len(img.shape)==2:
            # convert grayscale images to RGB
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # gif image
        elif len(img.shape)==1:
            img = img[0]
        # alpha channel image
        elif img.shape[2] == 4:
            img = img[:,:,0:3]

        img = Image.fromarray(img)
        
        if self.preprocessing_type == 0:
            # fixed resize classic Imagenet Preprocessing
            preprocessing = transforms.Compose([
                            transforms.Resize(self.fixed_size),                
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ]) 
        elif self.preprocessing_type == 1:
            # a bit smarter Imagenet preprocessing
            # at first resize by a smaller size, then do a center crop
            preprocessing = transforms.Compose([
                            transforms.Resize(self.fixed_size[0]),
                            transforms.CenterCrop(self.fixed_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])               
        elif self.preprocessing_type == 2:
            # fixed resize to a cluster-defined size
            preprocessing = transforms.Compose([
                            transforms.Resize(final_size),                
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])          
        elif self.preprocessing_type == 3:
            # some additional augmentations
            add_transforms = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                              RandomResizedCropRect(final_size,scale=(0.8, 1.0), ratio=(0.8, 1.2), interpolation=2),
                              ]

            preprocessing = transforms.Compose([
                            transforms.Resize(final_size),

                            transforms.RandomApply(add_transforms, p=self.prob),
                            transforms.RandomHorizontalFlip(p=self.prob),
                            transforms.RandomVerticalFlip(p=self.prob),
                            transforms.RandomGrayscale(p=self.prob),

                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])
        else:
            raise ValueError('This augmentation is not supported')

        img_arr = preprocessing(img).numpy()
        return img_arr    

class TelenavClassification(data.Dataset):
    def __init__(self,
                 mode = 'train', # 'train' or val'
                 random_state = 42,
                 fold = 0,
                 
                 size_ratio = 1.0,
                 preprocessing_type = 0, # 0,1,2
                 fixed_size = (1024,1024),
                 prob = 0.2,
                 
                 mean = (0.485, 0.456, 0.406),
                 std = (0.229, 0.224, 0.225),
                 
                 imgs_folder = '../../telenav/data/telenav_ai_dataset/train_data/',
                 df_path = '../data/telenav_df.feather',
                 return_img_id = False,
                 
                 all_classes = True,
                 multi_class = True
                ):
        
        self.std = std
        self.fold = fold
        self.mode = mode
        self.mean = mean        

        self.size_ratio = size_ratio
        self.fixed_size = fixed_size
        self.random_state = random_state
        
        self.prob = prob
        self.return_img_id = return_img_id
        self.preprocessing_type = preprocessing_type
        
        self.imgs_folder = imgs_folder
        self.df = pd.read_feather(df_path)
        self.all_classes = all_classes
        
        self.roi_hierarchy = {'GIVE_WAY':'OTHER_SIGN',
                         'SL_STOP_SIGN':'OTHER_SIGN',
                         'SL_US_10':'SPEED_LIMIT',
                         'SL_US_15':'SPEED_LIMIT',
                         'SL_US_20':'SPEED_LIMIT',
                         'SL_US_25':'SPEED_LIMIT',
                         'SL_US_30':'SPEED_LIMIT',
                         'SL_US_35':'SPEED_LIMIT',
                         'SL_US_40':'SPEED_LIMIT',
                         'SL_US_45':'SPEED_LIMIT',
                         'SL_US_5':'SPEED_LIMIT',
                         'SL_US_50':'SPEED_LIMIT',
                         'SL_US_55':'SPEED_LIMIT',
                         'SL_US_60':'SPEED_LIMIT',
                         'SL_US_65':'SPEED_LIMIT',
                         'SL_US_70':'SPEED_LIMIT',
                         'SL_US_75':'SPEED_LIMIT',
                         'SL_US_80':'SPEED_LIMIT',
                         'TRAFFIC_LIGHTS_SIGN':'TRAFFIC_LIGHTS',
                         'TURN_RESTRICTION_US_LEFT':'OTHER_SIGN',
                         'TURN_RESTRICTION_US_LEFT_UTURN':'OTHER_SIGN',
                         'TURN_RESTRICTION_US_RIGHT':'OTHER_SIGN',
                         'TURN_RESTRICTION_US_UTURN':'OTHER_SIGN'}        
        
        # ensure the same order
        self.type_list = sorted(self.df['roi_type'].unique())
        self.subtype_list = sorted(set([self.roi_hierarchy[_] for _ in self.type_list]))
        
        unq_df = pd.DataFrame(df[['img_name','bbox_size_bin']].groupby('img_name')['bbox_size_bin'].min())        
       
        self.stratify_values = list(unq_df.bbox_size_bin.values)
        self.filenames = list(unq_df.index.values)
        
        skf = StratifiedKFold(n_splits=5,
                              shuffle = True,
                              random_state = self.random_state)
        
        f1, f2, f3, f4, f5 = skf.split(self.filenames,
                                       self.stratify_values)
        
        folds = [f1, f2, f3, f4, f5]
        
        # no clusterization required
        if self.mode == 'train':
            self.train_idx = list(folds[self.fold][0])
        elif self.mode == 'val':
            self.val_idx = list(folds[self.fold][1])
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)
    def __getitem__(self, idx):
        if self.mode == 'train':
            img_index = self.train_idx[idx]
        elif self.mode == 'val':
            img_index = self.val_idx[idx]
        
        img_id = self.filenames[img_index]
        img_path = os.path.join(self.imgs_folder,img_id)
        
        roi_df = self.df[self.df.img_name==img_id]     
        
        w = roi_df.h.min()
        h = roi_df.w.min()
        x_crop = random.randint(0,w-1024)
        y_crop = random.randint(0,h-1024)
        crop_tuple = ((x_crop,x_crop+1024),(y_crop,y_crop+1024))
        crop_bbox = bb_tuple2dict_2d(crop_tuple)

        bboxes = []
        classes = []

        for i,row in roi_df.iterrows():
            bbox_tuple = ((row['tl_col'],row['br_col']),(row['tl_row'],row['br_row']))
            bboxes.append(bb_tuple2dict_2d(bbox_tuple))
            classes.append(row.roi_type)

        cropped_bboxes = []
        cropped_classes = []

        for bbox,class_ in zip(bboxes,classes):
            if bbox_inside_2d(bbox,crop_bbox):
                cropped_bboxes.append(bbox)
                cropped_classes.append(class_)
            else:
                if iou_2d(bbox,crop_bbox):
                    partial_bbox = {'x1': max(bbox['x1'],crop_bbox['x1']),
                                    'x2': min(bbox['x2'],crop_bbox['x2']),
                                    'y1': max(bbox['y1'],crop_bbox['y1']),
                                    'y2': max(bbox['y2'],crop_bbox['y2'])
                                   }
                    if iou_2d(bbox,partial_bbox)>0.5:
                        cropped_bboxes.append(partial_bbox)
                        cropped_classes.append(class_)               

        # one hot encode the classes
        # do it for classes / subclasses
        if self.all_classes:
            ohe_values = np.zeros(len(self.type_list))
            for _ in list(set(cropped_classes)):
                ohe_values[self.type_list.index(_)] = 1
        else:
            ohe_values = np.zeros(len(self.subtype_list))
            for _ in list(set(cropped_classes)):
                ohe_values[self.subtype_list.index(self.roi_hierarchy[_])] = 1            
        
        target_size = self.fixed_size
        img = self.preprocess_img(img_path,
                                  target_size,
                                  [slice(crop_bbox['y1'], crop_bbox['y2'], None),
                                   slice(crop_bbox['x1'], crop_bbox['x2'], None)])
        
        if img is None:
            # do not return anything
            pass
        else:
            # add failsafe values here
            
            if self.return_img_id == False:
                return_tuple = (img,
                                ohe_values)                
            else:
                return_tuple = (img,
                                ohe_values,
                                img_id)
            return return_tuple 
    def preprocess_img(self,
                       img_path,
                       target_size,
                       crop_slice
                       ):
            
        final_size =  [int(_ * self.size_ratio) for _ in target_size]
        img = imread(img_path)[crop_slice[0],crop_slice[1],:]
        
        # gray-scale img
        if len(img.shape)==2:
            # convert grayscale images to RGB
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # gif image
        elif len(img.shape)==1:
            img = img[0]
        # alpha channel image
        elif img.shape[2] == 4:
            img = img[:,:,0:3]

        img = Image.fromarray(img)
        
        if self.preprocessing_type == 0:
            # fixed resize classic Imagenet Preprocessing
            preprocessing = transforms.Compose([
                            transforms.Resize(self.fixed_size),                
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ]) 
        elif self.preprocessing_type == 1:
            # a bit smarter Imagenet preprocessing
            # at first resize by a smaller size, then do a center crop
            preprocessing = transforms.Compose([
                            transforms.Resize(self.fixed_size[0]),
                            transforms.CenterCrop(self.fixed_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])               
        elif self.preprocessing_type == 2:
            # fixed resize to a cluster-defined size
            preprocessing = transforms.Compose([
                            transforms.Resize(final_size),                
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])          
        elif self.preprocessing_type == 3:
            # some additional augmentations
            add_transforms = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                              RandomResizedCropRect(final_size,scale=(0.8, 1.0), ratio=(0.8, 1.2), interpolation=2),
                              ]

            preprocessing = transforms.Compose([
                            transforms.Resize(final_size),

                            transforms.RandomApply(add_transforms, p=self.prob),
                            transforms.RandomHorizontalFlip(p=self.prob),
                            transforms.RandomVerticalFlip(p=self.prob),
                            transforms.RandomGrayscale(p=self.prob),

                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])
        else:
            raise ValueError('This augmentation is not supported')

        img_arr = preprocessing(img).numpy()
        return img_arr    
    
class RandomResizedCropRect(transforms.RandomResizedCrop):
    """Extend the PyTorch function so that it could accept non-square images
    """    
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super(RandomResizedCropRect, self).__init__(size[0], scale, ratio, interpolation)
        self.size = size
        
def leval(x):
    return ast.literal_eval(x)      
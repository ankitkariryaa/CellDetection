import matplotlib.pyplot as plt
import os

import cv2
import numpy as np
import geopandas as gp
import rasterio
from sklearn.decomposition import PCA

from detection.utils import image_utils
from detection.utils.image_utils import get_annotated_img, filename_to_id
from detection.utils.logger import logger
from detection.utils.time_utils import time_it

np.random.seed(43)

gaussian_k = image_utils.gaussian_kernel((31, 31), 8, 1)


class FrameInfo():
    '''
    Contains annotated image information such as image data, annotations etc.
    '''

    def __init__(self, base_dir, img_id, roi, annotations, bbox=(15, 15)):
        self.base_dir = base_dir
        self.img_id = img_id
        # add buffer to region of interest ...
        self.roi = roi[0] - bbox[0], roi[1] + bbox[0], roi[2] - bbox[1], roi[3] + bbox[1]
        self.bbox = bbox
        # self.full_img = np.array(cv2.imread(os.path.join(self.base_dir, self.img_id)), dtype=np.float32)
        image = rasterio.open(os.path.join(self.base_dir, self.img_id))
        read_image = image.read()
        self.full_img = np.transpose(read_image, (1, 2, 0))
        self.img_data = self.full_img
        self.annotations = annotations
        self.all_seq_patches = []

    def annotated_img(self):
        annotated_img = get_annotated_img(self.img_data, self.annotations, self.bbox)
        return annotated_img

    def get_random_patches(self, patch_size, no_of_patches):
        '''
        Randomly samples no_of_patches of patch_size from image.
        '''
        img_shape = self.img_data.shape
        x = np.random.randint(0, img_shape[0] - patch_size[0], no_of_patches)
        y = np.random.randint(0, img_shape[1] - patch_size[1], no_of_patches)
        xy = zip(x, y)
        img_patches = []
        for i, j in xy:
            img_patch = Patch(self, j, i, patch_size)
            img_patches.append(img_patch)
        return img_patches

    def sequential_patches(self, patch_size, step_size):
        '''
        Returns all sequential patches from image separated by step.
        '''
        if len(self.all_seq_patches) == 0:
            img_shape = self.img_data.shape
            x = range(0, img_shape[0] - patch_size[0], step_size[0])
            y = range(0, img_shape[1] - patch_size[1], step_size[1])
            xy = [(i, j) for i in x for j in y]
            img_patches = []
            for i, j in xy:
                img_patch = Patch(self, j, i, patch_size)
                img_patches.append(img_patch)
            self.all_seq_patches = img_patches
        return self.all_seq_patches


class Patch(object):
    '''
    Represents a patch inside an input image.
    '''

    def __init__(self, frame_info, startx, starty, patch_size):
        self.frame_info = frame_info
        self.startx = startx
        self.starty = starty
        self.patch_size = patch_size
        self.__find_annotations()

    def get_img(self):
        img_data = self.frame_info.img_data
        img_patch = img_data[self.starty:self.starty + self.patch_size[0],
                    self.startx:self.startx + self.patch_size[1]]
        return img_patch

    def __find_annotations(self):
        '''
        Finds annotations whose bounding box completely lie in the patch.
        '''
        annotations = []
        for ann in self.frame_info.annotations:
            x, y, s = ann
            bbox_size = self.frame_info.bbox
            minx, miny, maxx, maxy = x - bbox_size[0], y - bbox_size[1], x + bbox_size[0], y + bbox_size[1]
            if minx >= self.startx and maxx <= self.startx + self.patch_size[1] and miny >= \
                    self.starty and maxy <= self.starty + self.patch_size[1]:
                # if self.startx <= x <= self.startx + self.patch_size[1] and self.starty <= y <= self.starty + \
                #         self.patch_size[1]:
                annotations.append(ann)
        self.ann_relative = annotations
        self.annotations = [(ann[0] - self.startx, ann[1] - self.starty, ann[2]) for ann in annotations]

    def annotated_img(self):
        ann_patch = get_annotated_img(self.get_img(), self.annotations, self.frame_info.bbox)
        return ann_patch

    def ann_mask(self, no_classes):
        img_mask = np.zeros(self.patch_size + (no_classes,))
        for ann in self.annotations:
            x, y, s = ann
            bbox = self.frame_info.bbox
            i = s if no_classes > 1 else 0
            if bbox[0] < x < self.patch_size[0] - bbox[0] and bbox[1] < y < self.patch_size[1] - bbox[1]:
                img_mask[y - bbox[1]:y + bbox[1] + 1, x - bbox[0]:x + bbox[0] + 1, i] = np.maximum(
                    img_mask[y - bbox[1]:y + bbox[1] + 1, x - bbox[0]:x + bbox[0] + 1, i], gaussian_k)
        return img_mask


class ImageDataset(object):
    '''
    Maintains training dataset and implements generators which provide data while training.

    '''

    def __init__(self, base_dir, file_suffix, annotation_file='localizations.txt', normalize=False):
        self.base_dir = base_dir
        self.file_suffix = file_suffix
        self.annotation_file = os.path.join(base_dir, annotation_file)
        self.all_frames = self.load_image_data()
        self.dataset_size = len(self.all_frames)
        if normalize:
            self.channel_mean = self.calc_channel_mean()
            self.normalize_frames()

    def get_frames(self):
        return self.all_frames

    def get_dataset_size(self):
        return self.dataset_size

    def load_image_data(self):
        '''
        Reads the annotation file and create frame objects for all image frames.
        '''
        logger.info('Loading data from directory:{}'.format(self.base_dir))
        all_annotations = {}

        all_files = os.listdir(self.base_dir)
        all_files = [fn for fn in all_files if fn.endswith(self.file_suffix)]
        all_files.sort(key=lambda x: filename_to_id(x))
        # since image files are listed sequentially in annotation file

        global_bounds_of_images = self.get_global_bounds_from_images(self.base_dir, all_files)

        trees = gp.read_file(self.annotation_file)
        trees.dropna(inplace=True)
        filtered_trees = trees[trees['geometry'].apply(lambda g: self.bound_contains_point(global_bounds_of_images, g))]
        # The ['KRONE_DM'] is divided by 0.20, as each pixel is 0.20 * 0.20 cm and KRONE_DM is in metres
        # For Sanju: The third parameter is the size right????
        filtered_trees_info = filtered_trees.apply(lambda row: (row['geometry'].x, row['geometry'].y, int(row['KRONE_DM']/0.2)), axis=1).values


        for (x1,y1,s) in filtered_trees_info:
            #TODO: Fix this hack when running on larger dataset!
            frame = int((x1 / 1000)-565)
            if frame not in all_annotations:
                all_annotations[frame] = []

            #  x1 and y1 are divided by 0.2 to convert to pixel index
            all_annotations[frame].append((int((x1 % 1000) / 0.2), int((y1 % 1000) / 0.2), s))

        roi = self.get_global_bounds(all_annotations)

        frame_infos = []
        total_annotations = 0
        for i, fn in enumerate(all_files):
            annotations = all_annotations[i] if i in all_annotations else []
            frame_info = FrameInfo(self.base_dir, fn, roi, annotations)
            frame_infos.append(frame_info)
            total_annotations += len(annotations)

        logger.info('Total frames loaded:{}, total annotations:{}', len(frame_infos), total_annotations)
        return frame_infos

    def bound_contains_point(self, bound, point):
        '''
        Reads the annotation file and create frame objects for all image frames.
        '''
        if (bound[0] < point.x and bound[2] >= point.x and bound[1] < point.y and bound[3] >= point.y):
            return True
        else:
            return False

    def get_global_bounds(self, all_annotations):
        '''
        Returns largest image region such that it covers complete annotated region in all input images.
        '''
        img_bounds = []
        for key, annotations in all_annotations.items():
            minx = min(annotations, key=lambda ann: ann[0])[0]
            maxx = max(annotations, key=lambda ann: ann[0])[0]
            miny = min(annotations, key=lambda ann: ann[1])[1]
            maxy = max(annotations, key=lambda ann: ann[1])[1]
            img_bounds.append((minx, maxx, miny, maxy))
        gminx = min(img_bounds, key=lambda x: x[0])[0]
        gmaxx = max(img_bounds, key=lambda x: x[1])[1]
        gminy = min(img_bounds, key=lambda x: x[2])[2]
        gmaxy = max(img_bounds, key=lambda x: x[3])[3]

        return gminx, gmaxx, gminy, gmaxy

    def get_global_bounds_from_images(self, base_dir, images):
        '''
        Returns largest image region such that it covers complete annotated region in all input images.
        '''
        img_bounds = []
        for image in images:
            print(base_dir + image)
            with rasterio.open(base_dir + image) as src:
                left, bottom, right, top = src.bounds
                img_bounds.append((left, bottom, right, top))
        gminx = min(img_bounds, key=lambda x: x[0])[0]
        gminy = min(img_bounds, key=lambda x: x[1])[1]
        gmaxx = max(img_bounds, key=lambda x: x[2])[2]
        gmaxy = max(img_bounds, key=lambda x: x[3])[3]

        return gminx, gmaxx, gminy, gmaxy

    def per_pixel_mean(self):
        img_frames = [frame.img_data for frame in self.all_frames]
        return np.mean(img_frames, axis=0)

    def calc_channel_mean(self):
        c1_frames = [frame.img_data[:, :, 0] for frame in self.all_frames]
        c2_frames = [frame.img_data[:, :, 1] for frame in self.all_frames]
        c3_frames = [frame.img_data[:, :, 2] for frame in self.all_frames]
        channel_mean = np.mean(c1_frames), np.mean(c2_frames), np.mean(c3_frames)
        logger.info('Channel mean:{}', channel_mean)
        return channel_mean

    def normalize_frames(self):
        for frame in self.all_frames:
            frame.img_data -= frame.img_data.mean() / (frame.img_data.std() + 1e-8)
        logger.info('Normalized frames with channel mean')

    def save_all_annotated(self, out_dir):
        for frame in self.all_frames:
            ann_frame = frame.img_data
            out_filename = os.path.join(out_dir, frame.img_id)
            cv2.imwrite(out_filename, ann_frame)

    @time_it
    def pixel_pca(self):
        pixel_vectors = []
        for frame in self.all_frames:
            pixel_vector = np.reshape(frame.img_data,
                                      [frame.img_data.shape[0] * frame.img_data.shape[1], frame.img_data.shape[2]])
            pixel_vectors.extend(pixel_vector.tolist())
        print('Total vectors', len(pixel_vectors))
        pca = PCA(n_components=3)
        pca.fit(np.array(pixel_vectors))
        cov = pca.get_covariance()
        print('Cov', cov)
        w, v = np.linalg.eig(cov)
        print(w, v)

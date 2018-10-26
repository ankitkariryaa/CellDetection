import os
import numpy as np
import geopandas as gps
import rasterio
from rasterio.transform import rowcol
import matplotlib.pyplot as plt

from detection.dataset.image_dataset import ImageDataset, FrameInfo


class TreeDataset(ImageDataset):
    def __init__(self, base_dir, file_suffix, annotation_file='trees.geojson', normalize=False):
        super(TreeDataset, self).__init__(base_dir, file_suffix, annotation_file, normalize)

    def load_image_data(self):
        all_files = os.listdir(self.base_dir)
        all_files = [fn for fn in all_files if fn.endswith(self.file_suffix)]
        gdf = gps.read_file(self.annotation_file)

        frame_infos = []
        total_annotations = 0
        for i, fn in enumerate(all_files):
            annotations = []
            image = rasterio.open(os.path.join(self.base_dir, fn))
            for _, row in gdf[gdf.image_idx == fn].iterrows():
                p = row['geometry']
                coord = list(p.coords[0])
                ann = rowcol(image.transform, coord[0], coord[1])
                if not np.isnan(row['STAMMUMFANG']):
                    tree_size = int(row['STAMMUMFANG'])
                else:
                    tree_size = 10
                tree_size = min(20, max(tree_size, 60))
                annotations.append((ann[1], ann[0], tree_size))

            frame_info = FrameInfo(self.base_dir, fn, (0, 0, image.shape[0], image.shape[1]), annotations)
            frame_infos.append(frame_info)
            total_annotations += len(annotations)
        return frame_infos


if __name__ == '__main__':
    dataset = TreeDataset('/home/sanjeev/Downloads/subset/', '.jpg', 'trees_out.geojson')
    for f in dataset.all_frames:
        img_out = f.annotated_img()
        plt.imshow(img_out)
        plt.show()

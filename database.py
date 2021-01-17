# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Segmentation
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
# Plot
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
import cv2

# Encoder - Decoder
# Texture
# sys.path.insert(0, '/home/giang.nguyen/FashionStyle/FashionPlus/generation')

# from options.train_options import TrainOptions
# from data.data_loader import CreateDataLoader
# from models.models import create_model
# import util.util as util
# import pickle


# # Shape
# sys.path.insert(1, '/home/giang.nguyen/FashionStyle/FashionPlus/separate_vae')


# Utils
import os
import time
from tqdm import tqdm 
import random
###############################

# %% [markdown]
# ```
# # label index:  segment name
# 0            :  null
# 1            :  hat
# 2            :  hair
# 3            :  glasses
# 4            :  up(per clothing)
# 5            :  skirt
# 6            :  pants
# 7            :  dress
# 8            :  belt
# 9            :  r-shoe
# 10           :  l-shoe
# 11           :  face
# 12           :  r-leg
# 13           :  l-leg
# 14           :  r-arm
# 15           :  l-arm
# 16           :  bag
# 17           :  scarf
# ```

# %%
class Retrieval:
    classes = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',                'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress',                'sling_dress', 'glasses', 'shoe', 'bag']
    mask_classes = ['up', 'up', 'up', 'up', 'up', 'up', 'pants', 'pants', 'skirt', 'dress',                     'dress', 'dress', 'dress', 'glasses', 'shoe', 'bag']
    seg_map = [4, 4, 4, 4, 4, 4, 6, 6, 5, 7, 7, 7, 7, 3, 9, 16]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    # threshold = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3]
    threshold = {'short_sleeved_shirt': 0.5, 'long_sleeved_shirt': 0.5, 'short_sleeved_outwear': 0.5, 'long_sleeved_outwear': 0.4,                'vest': 0.5, 'sling': 0.5, 'shorts': 0.5, 'trousers': 0.5, 'skirt': 0.5, 'short_sleeved_dress': 0.5, 'long_sleeved_dress': 0.5, 'vest_dress': 0.5,                'sling_dress': 0.5, 'glasses': 0.2, 'shoe': 0.4, 'bag': 0.3}

    model_df2_config = '/home/giang.nguyen/mmdetection/work_dirs/cascade_mask_rcnn_x101_64x4d_fpn_1x_df2/cascade_mask_rcnn_x101_64x4d_fpn_1x_df2.py'
    model_shoe_config = '/home/giang.nguyen/mmdetection/work_dirs/cascade_mask_rcnn_x101_64x4d_fpn_1x_sgb/cascade_mask_rcnn_x101_64x4d_fpn_1x_sgb.py'
    model_df2_pth = '/home/giang.nguyen/mmdetection/work_dirs/cascade_mask_rcnn_x101_64x4d_fpn_1x_df2/latest.pth'
    model_shoe_pth = '/home/giang.nguyen/mmdetection/work_dirs/cascade_mask_rcnn_x101_64x4d_fpn_1x_sgb/latest.pth'

    create_database = False

    mask_dest = 'datasets/humanparsing/train_mask'

    def __init__(self):
        # Load mmdet models
        self.model_shoe = init_detector(self.model_shoe_config, self.model_shoe_pth, device='cuda:0')
        self.model_df2 = init_detector(self.model_df2_config, self.model_df2_pth, device='cuda:0')

        # Load encoder

        # Load database
        if not self.create_database:
            # self.database = load()
            pass
        else:
            # self.database = create_database()
            pass
    
    def detect(self, image):
        file_name = image.split('/')[-1].replace('.jpg','.png')
        # print(file_name)
        tic = time.time()
        self.res_shoe = inference_detector(model=self.model_shoe, img=image)
        toc = time.time()
        self.res_df2  = inference_detector(model=self.model_df2, img=image)
        # print(self.res_shoe,'---', self.res_df2)   
        if len(np.array(self.res_shoe[0]).shape) > 1:
            tmp = np.random.rand(1,5)
            tmp[0,4] = 0.01
            self.res_shoe[0][0] = np.concatenate((self.res_shoe[0][0],tmp), axis = 0)
        if len(np.array(self.res_df2[0]).shape) > 1:
            tmp = np.random.rand(1,5)
            tmp[0,4] = 0.01
            self.res_df2[0][0] = np.concatenate((self.res_df2[0][0],tmp), axis = 0)

        self.res = np.concatenate((self.res_df2, self.res_shoe), axis=1)

    def draw_output(self, img, res, score=0.5):
        img = cv2.imread(img)
        res = self.res[0,:]
        # file_name = image.split('/')[-1]
        for i in range(len(res)):
            if len(res[i]) > 0 :
                for j in range(len(res[i])):
                    if res[i][j][4] > self.threshold[self.classes[i]]:
                        x1y1 = (res[i][j][0].astype(np.int32), res[i][j][1].astype(np.int32))
                        x4y4 = (res[i][j][2].astype(np.int32), res[i][j][3].astype(np.int32))

                        img = cv2.rectangle(img, x1y1, x4y4, self.colors[i], 2)
                        img = cv2.putText(img, '{} {:.4f}'.format(
                                        self.classes[i], res[i][j][4]),
                                        x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, self.colors[i], 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def create_database(self, image):
        file_name = image.split('/')[-1].replace('.jpg','.png')
        # self.masks = []
        self.mask = list()
        self.mask_items = dict()
        # self.mask_items = list()
        res = self.res[0,:]
        for i in range(len(res)):
            if len(res[i]) > 0 :
                for j in range(len(res[i])):
                    if res[i][j][4] > self.threshold[self.classes[i]]:
                        self.mask_items[self.mask_classes[i]] = 1*self.res[1,i][j]*self.seg_map[i]
                        # self.mask_items.append(1*self.res[1,i][j]*self.seg_map[i])
        self.mask = np.maximum.reduce([i for key, i in self.mask_items.items() if len(i) > 0])
        Image.fromarray(np.array(self.mask).astype(np.uint8)).save(f'{self.mask_dest}/{file_name}')
        # self.mask = np.maximum.reduce([i for i in self.mask_items if len(i) > 0])

    def load_database(self):
        pass

    def retrieaval(self, image):
        pass

    def plot_image(self, image):
        pass

demo = Retrieval()
print('Initialized')
    


# %%
source = 'datasets/humanparsing/train_img'

######################
files = os.listdir(source)

for img in tqdm(files):
    demo.detect(f'{source}/{img}')

    try:
        demo.create_database(f'{source}/{img}')
    except:
        print(img)


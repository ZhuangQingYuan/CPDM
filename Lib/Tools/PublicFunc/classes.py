from matplotlib import pyplot as plt
import numpy as np


CLASSES = {'pascal': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                      'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike',
                      'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'],

           'cityscapes': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                          'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                          'truck', 'bus', 'train', 'motorcycle', 'bicycle'],

           'coco': ['void', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
                    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge',
                    'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other',
                    'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
                    'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone',
                    'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other',
                    'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
                    'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow',
                    'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river',
                    'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
                    'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent',
                    'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other',
                    'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
                    'window-blind', 'window-other', 'wood'],
           'Potsdam':['impervious surface', 'building', 'low veg', 'tree', 'car'],

           'Vaihingen':['impervious surface', 'building', 'low veg', 'tree', 'car'],

           'MSL':['Martian Soil', 'Sands', 'Gravel', 'Bedrock', 'Rocks','Tracks', 'Shadows', 'Unknown', 'Background'], # 9 类

           'MER':['Martian Soil', 'Sands', 'Gravel', 'Bedrock', 'Rocks','Tracks', 'Shadows', 'Unknown', 'Background'],# 9 类

           'DFC22':['Urban fabric', 'Industrial', 'Mine', 'Artificial', 'Arable', 'Permanent crops','Pastures', 'Forests', 'Herbaceous', 'Open spaces', 'Wetlands', 'Water'],# 12 类

           'iSAID':['Ship', 'Storage_Tank', 'Baseball_Diamond', 'Tennis_Court', 'Basketball_Court','Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter','Swimming_Pool', 'Roundabout','Soccer_Ball_Field', 'Plane', 'Harbor'], #15类

           'GID15':['industrial_land', 'urban_residential', 'rural_residential', 'traffic_land', 'paddy_field','irrigated_land', 'dry_cropland', 'garden_plot', 'arbor_woodland', 'shrub_land','natural_grassland', 'artificial_grassland', 'river', 'lake', 'pond'] # 15类

           }


TSNE_VAIHINGEN_COLORMAP = {
            'impervious surface' : np.array([255, 255, 255]),  # label 0
            'building' : np.array([0, 0, 255]),  # label 1
            'low veg' : np.array([0, 255, 255]),  # label 2
            'tree' : np.array([0, 255, 0]),  # label 3
            "car" : np.array([255, 255, 0]),  # label 4
        }

TSNE_DFC22_COLORMAP = {
    'Urban fabric': np.array([255, 255, 255]),      # 鲜红色
    'Industrial': np.array([0, 0, 255]),        # 鲜绿色
    'Mine': np.array([0, 255, 255]),            # 亮黄色
    'Artificial': np.array([0, 255, 0]),        # 亮蓝色
    'Arable': np.array([255, 255, 0]),           # 橙色
    'Permanent crops': np.array([145, 30, 180]),  # 紫色
    'Pastures': np.array([70, 240, 240]),         # 青色
    'Forests': np.array([240, 50, 230]),          # 粉紫色
    'Herbaceous': np.array([210, 245, 60]),       # 黄绿色
    'Open spaces': np.array([250, 190, 190]),     # 粉红色
    'Wetlands': np.array([0, 128, 128]),          # 蓝绿色
    'Water': np.array([0, 0, 128])                # 深蓝色
}

TSNE_ISAID_COLORMAP ={

} 

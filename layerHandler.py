from statistics import *

from mmdet.apis import init_detector, inference_detector
from mmdet.structures.mask import bitmap_to_polygon

from scaleHandler import *


def init_layer_model():
    # Specify the path to model config and checkpoint file
    config_file = './models/layer-inference/config-layer.py'
    checkpoint_file = './models/layer-inference/epoch_36.pth'

    # Build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cpu')

    return model


def detect_layer(img,model):
    result = inference_detector(model, img)
    return result

def process_layer_result(result):
    polygon = bitmap_to_polygon(result.pred_instances.masks[0])
    x = []
    y = []
    layer_coords = []
    Xmax = result.ori_shape[1]
    for i in polygon[0][0]:
        if (i[0] < Xmax * 0.94) & (i[0] > Xmax * 0.04):
            x.append(i[0])
            y.append(i[1])
            layer_coords.append([i[0], i[1]])
    Yavg = sum(y) / len(y)
    XYtop = []
    XYlow = []
    for point in layer_coords:
        if point[1] < Yavg:
            XYtop.append(point)
        else:
            XYlow.append(point)

    XYtop = sorted(XYtop, key=lambda x: x[0])
    XYlow = sorted(XYlow, key=lambda x: x[0])

    x_top, y_top = zip(*XYtop)
    top_approx = np.polyfit(x_top, y_top, 1)
    x_low, y_low = zip(*XYlow)
    low_approx = np.polyfit(x_low, y_low, 1)
    TopEdgeMean = round(np.mean(y_top), 0)
    LowEdgeMean = round(np.mean(y_low), 0)
    distances=[]
    for i in range(len(XYtop)):
            if (XYtop[i][0]==XYlow[i][0]):
                distances.append(math.hypot(XYtop[i][0] - XYlow[i][0], XYtop[i][1] - XYlow[i][1]))
            if(XYlow[i][0]==XYlow[-1][0]):
                break
    return x_top, y_top, x_low, y_low, top_approx, low_approx, max(distances), median(distances), min(distances), PolyArea(layer_coords);


def calculate_coat_thickness(img, px_thickness, scale_length, scale_value):
    area_in_pixels = img.shape[0] * img.shape[1]
    print(img.shape[0],scale_length,scale_value)
    height_in_micrometers = img.shape[0] / scale_length * scale_value
    width_in_micrometers = img.shape[1] / scale_length * scale_value

    area_in_micrometers = height_in_micrometers * width_in_micrometers

    median_coat_thickness = px_thickness * height_in_micrometers / img.shape[0]
    return median_coat_thickness

def PolyArea(poly):
    import numpy as np
    x=[]
    y=[]
    x,y=zip(*poly)
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


import torch
import torch.nn as nn

from utils import *
from yolov3_tiny import *
from ptflops import get_model_complexity_info

os.environ["CUDA_VISIBLE_DEVICES"]="6"

model = Yolov3Tiny(num_classes=80)
# with torch.cuda.device(0):
#   macs, params = get_model_complexity_info(model, (3, 416, 416), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))

model.load_state_dict(torch.load('yolov3_tiny.h5'))
imgfile = "test_img.jpg"

img_org = Image.open(imgfile).convert('RGB')
img_resized = img_org.resize((416, 416))
img_torch = image2torch(img_resized)

all_boxes = model.predict_img(img_torch, conf_thresh=0.2)[0]
boxes = nms(all_boxes, 0.4)
# print(len(boxes))
plot_img_detections(img_resized, boxes, figsize=(8,8), class_names=class_names)
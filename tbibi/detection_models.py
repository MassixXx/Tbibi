import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image






def predict_skin_illness(img):

    graph_def = tf.compat.v1.GraphDef()
    labels = []

    # These are set to the default names from exported models, update as needed.
    filename = "models/model.pb"
    labels_filename = "models/labels.txt"

    # Import the TF graph
    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # Create a list of labels.
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())


    # Open the image to test
    imageFile = img
    image = Image.open(imageFile)

    # Update orientation based on EXIF tags, if the file has orientation info.
    image = update_orientation(image)

    # Convert to OpenCV format
    image = convert_to_opencv(image)

    # If the image has either w or h greater than 1600 we resize it down respecting
    # aspect ratio such that the largest dimension is 1600
    image = resize_down_to_1600_max_dim(image)

    # We next get the largest center square
    h, w = image.shape[:2]
    min_dim = min(w, h)
    max_square_image = crop_center(image, min_dim, min_dim)

    # Resize that square down to 256x256
    augmented_image = resize_to_256_square(max_square_image)

    # Get the input size of the model
    with tf.compat.v1.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
    network_input_size = input_tensor_shape[1]

    # Crop the center for the specified network_input_Size
    augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

    # These names are part of the model and cannot be changed.
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'

    with tf.compat.v1.Session() as sess:
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions = sess.run(prob_tensor, {input_node: [augmented_image]})
            predictions[0,3] = 0
            predictions[0,5] = 0
            
            skin = False
            for p in predictions[0]:
              if p>0.3:
                skin = True
                break
            
        
                    
                   
                # Print the highest probability label
            highest_probability_index = np.argmax(predictions)
            str = 'Classified as: ' + labels[highest_probability_index]
            return {
                "detected" : True,
                "type" : str,
                "degree":0, #'cause it's not a burn
                "emergency" : False
            } if skin else {
            			"detected":False,
            			"type":None,
            			"degree":False,
            			"emergency":False
    			}

        except KeyError:
                    print("Couldn't find classificaqtion output layer: " + output_layer + ".")
                    print("Verify this a model exported from an Object Detection project.")
                    exit(-1)
##################*********************************************************************##################

import argparse
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
opt = {
    "weights" : 'skin_burn_2022_8_21.pt',
    "source" : 'images',
    "img_size" : 640,
    "conf_thres" : 0.45,
    "iou_thres" : 0.15,
    "device" : '',
    "view_img" : None,
    "save_txt" : None,
    "save_conf" : None,
    "nosave" : None,
    "classes" : None,
    "agnostic_nms" : None,
    "augment" : None,
    "update" : None,
    "project" : 'runs/detect',
    "name" : 'exp',
    "exist_ok" : None,
    "no_trace" : None
}


source, weights, view_img, save_txt, imgsz, trace = opt['source'], opt['weights'], opt['view_img'], opt['save_txt'], opt['img_size'], not opt['no_trace']
save_img = not opt['nosave'] and not source.endswith('.txt')  # save inference images
webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://', 'https://'))

# Directories
save_dir = Path(increment_path(Path(opt['project']) / opt['name'], exist_ok=opt['exist_ok']))  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Initialize
set_logging()
device = select_device(opt['device'])
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, opt['img_size'])

if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()



# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1



def predict_burn(image_path):
      detected = False
      degree = 0
      dataset = LoadImages(image_path, img_size=imgsz, stride=stride)
      t0 = time.time()
      for path, img, im0s, vid_cap in dataset:
          img = torch.from_numpy(img).to(device)
          img = img.float()  # uint8 to fp16/32
          img /= 255.0  # 0 - 255 to 0.0 - 1.0
          if img.ndimension() == 3:
              img = img.unsqueeze(0)
      # Warmup
      if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
          old_img_b = img.shape[0]
          old_img_h = img.shape[2]
          old_img_w = img.shape[3]
          for i in range(3):
              model(img, augment=opt['augment'])[0]

    # Inference
      t1 = time_synchronized()
      pred = model(img, augment=opt['augment'])[0]
      t2 = time_synchronized()

      # Apply NMS
      pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
      t3 = time_synchronized()

      # Apply Classifier
      if classify:
          pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
      for i, det in enumerate(pred):  # detections per image
          if webcam:  # batch_size >= 1
              p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
          else:
              p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

          p = Path(p)  # to Path
          save_path = str(save_dir / p.name)  # img.jpg
          txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
          gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
          if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                detected = True
                n = (det[:, -1] == c).sum()  # detections per class
                degree = int(names[(int(c))][0])
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

              # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt['save_conf'] else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
          print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

          # Stream results
          if view_img:
              cv2.imshow(str(p), im0)
              cv2.waitKey(1)  # 1 millisecond

          # Save results (image with detections)
          if save_img:
              if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
              

          if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
      #print(f"Results saved to {save_dir}{s}")

      print(f'Done. ({time.time() - t0:.3f}s)')

      return {
          "detected" : detected,
          "degree" : degree
      }

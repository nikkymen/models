import numpy as np
import os
import urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import scipy.misc

from urllib.request import URLopener

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/nik/Work/wega/tf_detection/faster_rcnn_inception_resnet_v2_car_1.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/nik/Work/wega/tf_detection/train/car_label_map.pbtxt'

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ '/home/kozhanov/projects/tensorflow/val_data/car_1/images/Автомобиль легковой/000/00000081_co_0088_0000_0.jpeg' ]

with open('/home/nik/Work/wega/tf_detection/val_data/car_1/image_list.txt') as f:
  TEST_IMAGE_PATHS = f.read().splitlines()

TEST_IMAGE_PATHS = ['/home/nik/Work/wega/tf_detection/val_data/car_1/images/Автомобиль легковой/000/00000039_co_0002_0000_0.jpeg']

# Size, in inches, of the output images.
IMAGE_SIZE = (8, 8)

session_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)

# session_config.gpu_options.allow_growth = True

# session_config.gpu_options.per_process_gpu_memory_fraction = 0.05

with detection_graph.as_default():
  with tf.Session(graph=detection_graph, config=session_config) as sess:

    #train_writer = tf.summary.FileWriter('/home/nik/Work/wega/faster_rcnn_resnet101_coco_11_06_2017_graph')
    #train_writer.add_graph(sess.graph)

    for idx, image_path in enumerate(TEST_IMAGE_PATHS):
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      rpn_conv_weights = detection_graph.get_tensor_by_name('Conv/weights:0')


      # Actual detection.

 #     graph_ops = detection_graph.get_operations()

      # with open('graph.txt', 'w') as f:
      #     print(detection_graph.get_operations(), file=f, sep='\n')
          # for line in graph_ops:
          #     f.write("%s\n" % line)

     # print(detection_graph.get_operations(), file=f, sep='\n')

      #print(image_np_expanded.flatten())

      print('rpn_conv_weights', rpn_conv_weights)

      start = time.perf_counter()

      (boxes, scores, classes, num_detections, rpn_conv_weights) = sess.run(
          [boxes, scores, classes, num_detections, rpn_conv_weights],
          feed_dict={image_tensor: image_np_expanded},
          )

      # print("boxes shape", boxes.shape)
      # print("scores shape", scores.shape)
      # print("classes shape", classes.shape)
      # print("num_detections shape", num_detections.shape)
      print("scores:", scores)

      print(image_path, 'elapsed: ', time.perf_counter() - start)

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=2,
          max_boxes_to_draw=None)

      #scipy.misc.imsave('/home/nik/Work/wega/tf_detection/tests/car_1_test_1/{}.png'.format(idx), image_np)
      scipy.misc.imsave('/home/nik/Work/wega/tf_detection/tests/test.png', image_np)
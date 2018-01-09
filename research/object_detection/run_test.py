import numpy as np
import os
import urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import scipy.misc
import pickle

from urllib.request import URLopener

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz (300x300)
# http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

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
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 5) ]
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test.jpg') ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
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
      # Actual detection.

      resize_layer = detection_graph.get_tensor_by_name('Preprocessor/map/while/ResizeToRange/ResizeBilinear:0')

      print('resize_layer:', resize_layer)

      lst = detection_graph.get_operations()

      with open('graph.txt', 'w') as f:
       print(lst, file=f,sep='\n')

      first_conv_layer = detection_graph.get_tensor_by_name('FirstStageFeatureExtractor/InceptionResnetV2/InceptionResnetV2/Conv2d_1a_3x3/convolution:0')
      last_conv_layer = detection_graph.get_tensor_by_name('FirstStageFeatureExtractor/InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Conv2d_1x1/convolution:0')

      # size_tensor = detection_graph.get_tensor_by_name('Preprocessor/map/while/ResizeImage/size:0')
      # resized_tesnor = detection_graph.get_tensor_by_name('Preprocessor/map/while/ResizeImage/ResizeBilinear:0')


      start = time.perf_counter()

      (boxes, scores, classes, num_detections, first_conv_mat, last_conv_mat, resized_mat) = sess.run(
          [boxes, scores, classes, num_detections, first_conv_layer, last_conv_layer, resize_layer],
          feed_dict={image_tensor: image_np_expanded})

      print('resized_mat ', resized_mat.shape)
      print('first_conv_mat ', first_conv_mat.shape)
      print('last_conv_mat ', last_conv_mat.shape)
    #  scipy.misc.imsave('outfile.png', resized_image[0])

     # print(image_path, 'elapsed: ', time.perf_counter() - start)

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      scipy.misc.imsave('outfile.png', image_np)

      # plt.figure(figsize=IMAGE_SIZE)
      # plt.imshow(image_np)
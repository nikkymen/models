import os
import json
import PIL.Image
import io
import hashlib
import numpy

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags

flags.DEFINE_string('db_path', None, 'Path to dpf database.')
flags.DEFINE_bool('mask', None, 'Write masks.')

FLAGS = flags.FLAGS


def translit(string):
    symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
               u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA")

    tr = {ord(a): ord(b) for a, b in zip(*symbols)}
    return string.translate(tr)


def str_to_point(string):
    pt = str.split(string, ';')
    return [float(pt[0]), float(pt[1])]


def main(_):

    image_dir = FLAGS.db_path

    label_map_dict = json.load(open(image_dir + '/labels.json'))

    # generate label_map.pbtxt

    label_map_proto = label_map_util.string_int_label_map_pb2.StringIntLabelMap()

    for key, val in label_map_dict.items():
        item = label_map_proto.item.add()
        item.id = val
        item.name = key

    with open(image_dir + '/label_map.pbtxt', 'w') as f:
        f.write("{}".format(label_map_proto))

    ##

    writer = tf.python_io.TFRecordWriter(image_dir + '/train.record')

    with open(image_dir + '/image_list.txt') as f:
        image_list = f.read().splitlines()

    for idx, image_path in enumerate(image_list):

       # label_prefix = os.path.abspath(os.path.join(image_dir, '../' + image_path))

        image_path = os.path.normpath(image_path)

        label_prefix = os.path.splitext(image_path)[0]
        label_prefix = label_prefix.split(os.sep)[1:]
        label_prefix = os.path.join('labels', os.sep.join(label_prefix))
        label_prefix = os.path.join(image_dir, label_prefix)

        # image_list contains RELATIVE path
        image_path = os.path.join(image_dir, image_path)

        annotation_path = os.path.join(image_dir, os.path.splitext(image_path)[0] + '.json')
        with open(annotation_path) as data_file:
            json_node = json.load(data_file)

        with tf.gfile.GFile(image_path, 'rb') as fid:
            encoded_image = fid.read()

        image = PIL.Image.open(io.BytesIO(encoded_image))

        key = hashlib.sha256(encoded_image).hexdigest()
        width, height = image.size

        filename = os.path.basename(image_path).encode()  # Filename of the image. Empty if image is not from file
        image_format = b'jpeg'

        xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
        ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
        classes_text = []  # List of string class name of bounding box (1 per box)
        classes = []  # List of integer class id of bounding box (1 per box)
        masks = []  # List of PNG-encoded arrays

        for target in json_node['targets']:

            index = len(classes)

            target_string_id = target['class_id']
            target_id = label_map_dict[target_string_id]

            target_string_id = translit(target_string_id)

            target_poly_string = target['polygon'][2:-2]

            points_arr = str.split(target_poly_string, ');(')

            points_arr = [str_to_point(i) for i in points_arr]

            points_arr = numpy.array(points_arr)

            minx = points_arr[:, 0].min()
            miny = points_arr[:, 1].min()
            maxx = points_arr[:, 0].max()
            maxy = points_arr[:, 1].max()

            minxr = max(0, minx / width)
            minyr = max(0, miny / height)

            maxxr = min(1, maxx / width)
            maxyr = min(1, maxy / height)

            xmins.append(minxr)
            ymins.append(minyr)
            xmaxs.append(maxxr)
            ymaxs.append(maxyr)

            classes.append(target_id)
            classes_text.append(target_string_id.encode('utf8'))

            if FLAGS.mask:
                mask_image_path = os.path.join(label_prefix, str(index) + '.png')

                with tf.gfile.GFile(mask_image_path, 'rb') as fid:
                    mask_array = fid.read()

                masks.append(mask_array)

        if classes:
            example_dict = {
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_image),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }

            if FLAGS.mask:
                example_dict['image/object/mask'] = (dataset_util.bytes_list_feature(masks))

            tf_example = tf.train.Example(features=tf.train.Features(feature=example_dict))

            writer.write(tf_example.SerializeToString())

        print('{}/{} completed'.format(idx + 1, len(image_list)))

    writer.close()

if __name__ == '__main__':
    tf.app.run()
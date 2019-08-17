import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections # 對字典元素做排序的package
import re
import cv2
import random
from tqdm import tqdm # 用來顯示進度條
import time
import scipy.misc # 用cv2或skimage(讀取稍微比cv2快, 但是resize所需時間十倍長於cv2)代替吧
import numpy as np
from scipy.stats import truncnorm # 也可以用tf.truncated_norm代替


def get_truncated_normal(mean=0, sd=0.5):
    return truncnorm((sd*-2 - mean) / sd, (sd*2 - mean) / sd, loc=mean, scale=sd)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True) # 分析變數大小


def get_assigment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    '''param assigment: 要回復的變數'''
    initialized_variable_names = {}
    name_to_variable = collections.OrderedDict() # 創建一個有序的字典對象
    
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
        
    '''補充說明: 返回讀取的已保存參數的值: tf.train.load_variable()。'''
    # 將已保存參數的（名稱，形狀）以列表的形式返回。
    init_vars = tf.train.list_variables(init_checkpoint)
    
    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


# loading weights
def init_from_checkpoint(init_checkpoint):
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = get_assigment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    for t in initialized_variable_names:
        if ":0" not in t:
            tf.logging.info("loading weights success: " + t)


def get_tfrecords(img_paths, output_dir, img_size):
    random.shuffle(
        img_paths)  # As tfrecords only can shuffle in small buffer size, we need to shuffle the total data before generate the tfrecord
    writer = tf.python_io.TFRecordWriter(output_dir)
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size[0], img_size[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={"img": tf.train.Feature(bytes_ytesList(valulist=tf.train.Be=[img.tostring()]))}))
        writer.write(example.SerializeToString())
    writer.close()


def input_fn(input_file, batch_size, img_size, buffer_size=10000):
    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, features={"img": tf.FixedLenFeature([], tf.string)})
        img = tf.decode_raw(example["img"], tf.uint8)
        img = tf.reshape(img, img_size)

        return img

    """The actual input function."""
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    d = d.repeat()
    d = d.shuffle(buffer_size=buffer_size)

    d = d.apply(
        tf.contrib.data.map_and_batch(lambda record: _decode_record(record),
                                      batch_size=batch_size, drop_remainder=True))
    return d


def cal_ETA(t_start, i, n_batch):
    t_temp = time.time()
    t_avg = float(int(t_temp) - int(t_start)) / float(i + 1) # 經過多久 / 步數 = 平均花多久
    if n_batch - i - 1 > 0:
        return int((n_batch - i - 1) * t_avg) # 還有多少batch * t_avg = 還剩多久時間
    else:
        return int(t_temp) - int(t_start)


def merge(images, size):
    # 檢查傳進來的batch有幾張圖, 若超多張圖(大於size[0] * size[1])就限制住不能超過它(size[0] * size[1])
    if images.shape[0] > size[0] * size[1]:
        images = images[0:size[0] * size[1], ::]
        
    # height, width  
    h, w = images.shape[1], images.shape[2]
    
    # 檢查通道數是否合法(RGB: 3  or RGBA: 4)
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        # 創建全黑大圖
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            # i: col, j: row(每幾個才會跳一次)
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) / 2.


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

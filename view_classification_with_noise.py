"""
training and evaluation pipeline for echocardiography view classification.

This file contains: data loading, augmentation, contrastive pretraining,
label-noise correction, downstream evaluation, and reporting helpers.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf
import keras_cv
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import numpy as np
import sys
import random
import math
import cv2
import itertools
#import pydevd
import tensorflow_addons as tfa
import pandas as pd
import shutil
import csv
import time
import enum
import json
import argparse
from collections import Counter
from collections import defaultdict
from typing import List, Dict, Tuple
from matplotlib.colors import to_rgb
import colorsys

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import inspect
from keras_cv_attention_models import attention_layers as A

import utils
from experts_analysis import get_expert_labels
from evaluation_analysis import *
from experiment_plots import *
from statistical_significance_test import *

IMAGE_SIZE = (224, 224, 3)
SSL_BATCH_SIZE = 64 #64 #128#64
BATCH_SIZE = 64
EXCLUDE_CLASS_LESS_THAN_SAMPLES = 200#10
NUM_CLASSES = 47#53

TEMPERATURE = 0.1
NETWORK_WIDTH = 128
DROP_K = 1
DROP_V = 3

#Too small a temperature (e.g., tau < 0.03) can lead to gradient instability or overfitting to training pairs.
#Empirical sweet spots are often:
#Unsupervised contrastive: tau in [0.1, 0.5]
#Supervised contrastive: tau in [0.05, 0.2]

LOSSES = ['supcon',
          'dropcon',
          'compcon',
          'compcon_hybrid',
          'supcon_softscale',
          'supcon_sqmax',
          'supcon_weighted',
          'logsum',
          'logsum_weighted',
          'logsum_expavgsum',
          'logsum_neg_emph',
          'hybrid',
          'pairwise',
          'dcl_softmax',
          'sup_barlow',
          'supcon_adaptive']
USE_LOSS = LOSSES[0]

BACKBONES = [
    "xception",
    "resnet50",
    "resnet101",
    "densenet121",
    "convnextbase",
    "efficientnetv2s",
    "vit_base",
    "swintransformerv2base",
    "convnexttiny",
    "vit_small",
    "swintransformerv2tiny",

]
USE_BACKBONE = BACKBONES[0]

#These groupings are used for structured label-noise injection.
groupings_dict = {"apical": {'a2ch-la': 1, 'a2ch-lv': 2, 'a2ch-full': 3, 'apex': 4,
                             'a3ch-lv': 1, 'a3ch-la': 2, 'a3ch-full': 3, 'a3ch-outflow': 4,
                             'a4ch-lv': 1, 'a4ch-la': 2, 'a4ch-full': 3, 'a4ch-ias': 4, 'a4ch-rv': 5, 'a4ch-ra': 6,
                             'a5ch-full': 1, 'a5ch-outflow': 2},
                  "plax": {'plax-full-out': 1, 'plax-full-lv': 2, 'plax-full-la': 3, 'plax-full-rv-ao': 4,
                           'plax-full-mv': 5, 'plax-valves-av': 6, 'plax-valves-mv': 7, 'plax-tv': 8},
                  #'plax-valves-pv': 9,
                  "psax": {'psax-all': 1, 'psax-av': 2, 'psax-tv': 3, 'psax-pv': 4,
                           'psax-lv-base': 1, 'psax-lv-mid': 2, 'psax-lv-apex': 3},
                  "sub": {'subcostal-heart': 1, 'subcostal-ivc': 2, 'suprasternal': 3},
                  "mmode": {'mmode-a4ch-rv': 1, 'mmode-ivc': 2, 'mmode-plax-mitral': 3, 'mmode-plax-av': 4,
                            'mmode-plax-lv': 5},
                  "doppler": {'doppler-ao-descending': 1, 'doppler-tissue-lateral': 2, 'doppler-mv': 3, 'doppler-av': 4,
                              'doppler-tissue-septal': 5, 'doppler-pv': 6, 'doppler-tissue-rv': 7, 'doppler-tv': 8},
                  }

def plot_confusion_matrix_2(cm,
                      target_names,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions


    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    FONT_SIZE = 8

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8*2, 6*2))    #8, 6
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=FONT_SIZE)
        plt.yticks(tick_marks, target_names, fontsize=FONT_SIZE)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=FONT_SIZE,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=FONT_SIZE,
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label (accuracy={:0.4f}; misclass={:0.4f})'.format(accuracy, misclass))
    plt.show()

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements an LR scheduler that warms up the learning rate for some training steps
    (usually at the beginning of the training) and then decays it
    with CosineDecay (see https://arxiv.org/abs/1608.03983)
    """

    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

def split_data(data_folder, perc_train, perc_val, perc_test, seed = 444, files_for_relabelling=None, dual_files_dict=None):
    dirs = os.listdir(data_folder)
    dirs.sort()

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_files, test_labels = [], []
    class_count = {}
    class_count_train = {}
    class_count_val = {}
    class_count_test = {}
    class_lookup = {}
    class_label = 0
    num_excluded = 0
    excluded = {}
    num_excluded_dual_per_class = {}

    for dir in dirs:

        if dir.lower() == 'unclassified' or dir.lower() == 'not-sure':
            continue

        files = os.listdir(os.path.join(data_folder, dir))
        num_files = len(files)
        if num_files < EXCLUDE_CLASS_LESS_THAN_SAMPLES:
            num_excluded+=1
            excluded[dir] = len(files)
            continue

        files_with_path = []
        for file in files:
            files_with_path.append(os.path.join(os.path.join(data_folder, dir), file))

        if files_for_relabelling is not None:
            temp_files_with_path = []
            for i in range(len(files_with_path)):
                if os.path.basename(files_with_path[i]) not in files_for_relabelling:
                    temp_files_with_path.append(files_with_path[i])
            files_with_path = temp_files_with_path

        dual_cnt = 0
        if dual_files_dict is not None:
            temp_files_with_path = []
            for img_file in files_with_path:
                if img_file not in dual_files_dict:
                        temp_files_with_path.append(img_file)
                else:
                    dual_cnt += 1
            files_with_path = temp_files_with_path
        num_excluded_dual_per_class[dir] = dual_cnt

        num_files = len(files_with_path)
        num_train = int(num_files * perc_train)
        num_val = int(num_files * perc_val)
        num_test = int(num_files * perc_test)

        random.seed(seed)
        random.shuffle(files_with_path)

        test = files_with_path[0 : num_test]
        val = files_with_path[num_test : num_test + num_val]
        train = files_with_path[num_test + num_val : num_test + num_val + num_train]

        labels_train = [class_label] * num_train
        labels_val = [class_label] * num_val
        labels_test = [class_label] * num_test

        train_files.extend(train)
        train_labels.extend(labels_train)
        val_files.extend(val)
        val_labels.extend(labels_val)
        test_files.extend(test)
        test_labels.extend(labels_test)

        class_count[dir] = len(files)
        class_count_train[dir] = len(train)
        class_count_val[dir] = len(val)
        class_count_test[dir] = len(test)
        class_lookup[class_label] = dir
        class_label += 1

    temp = list(zip(train_files, train_labels))
    random.shuffle(temp)
    train_files, train_labels = zip(*temp)

    temp = list(zip(val_files, val_labels))
    random.shuffle(temp)
    val_files, val_labels = zip(*temp)

    temp = list(zip(test_files, test_labels))
    random.shuffle(temp)
    test_files, test_labels = zip(*temp)

    #Uncomment to make it an exact multiple of batch size
    #per_step = len(train_files) / float(BATCH_SIZE)
    #num_train = math.floor(per_step) * BATCH_SIZE
    #train_files = train_files[:num_train]
    #train_labels = train_labels[:num_train]
    #per_step = len(val_files) / float(BATCH_SIZE)
    #num_val = math.floor(per_step) * BATCH_SIZE
    #val_files = val_files[:num_val]
    #val_labels = val_labels[:num_val]

    print(f'NUM EXCLUDED: {num_excluded}')

    return (list(train_files), list(train_labels),
            list(val_files), list(val_labels),
            list(test_files), list(test_labels),
            class_count, class_lookup, class_count_train, class_count_val, class_count_test,
            num_excluded_dual_per_class)

def process_image_only(image_file):
    ext = tf.strings.substr(image_file, tf.strings.length(image_file) - 3, 3)
    png_txt = tf.convert_to_tensor('png')
    jpg_txt = tf.convert_to_tensor('jpg')
    #Pydevd.settrace(suspend=False)
    img = tf.io.read_file(image_file)

    if (tf.strings.regex_full_match(ext, png_txt)):
        image = tf.image.decode_png(img, channels=3)
    else:  #(tf.strings.regex_full_match(ext, jpg_txt)):
        image = tf.image.decode_jpeg(img, channels=3)

    #image = tf.image.rgb_to_grayscale(image)

    h, w = image.shape[:2]

    if h!= IMAGE_SIZE[0] and w!= IMAGE_SIZE[1]:
        image = tf.image.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

    return image

def process_image_and_label(image_file, label):
    return process_image_only(image_file), label


def get_augmenter_echo(image_size, min_area, rotation):
    zoom_factor = 1.0 - math.sqrt(min_area)
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=image_size),
            tf.keras.layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            tf.keras.layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            tf.keras.layers.RandomRotation(rotation)
        ]
    )

def get_tf_datasets(train_files, train_labels,
                     val_files, val_labels,
                     test_files, test_labels, augmentation = False):

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.map(process_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 10).batch(BATCH_SIZE)

    if augmentation:
        AUG_MIN_AREA = 0.8
        AUG_ROTATION = 0.10
        print('AUGMENTATION ENABLED.')
        aug = get_augmenter_echo(IMAGE_SIZE, AUG_MIN_AREA, AUG_ROTATION)
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(process_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(buffer_size=BATCH_SIZE * 10).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_ds = test_ds.map(process_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_images_ds = tf.data.Dataset.from_tensor_slices((test_files))
    test_images_ds = test_images_ds.map(process_image_only, num_parallel_calls=tf.data.AUTOTUNE)
    test_images_ds = test_images_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, test_images_ds

def clean_files(files_list):
    '''
    A few image files (around 8 or so) were corrupted, but it is hard to find out which ones when it fails
    inside a tensorflow mapped function. Run this once to clean the dataset and fix this problem.
    '''
    cnt_removed = 0
    for file in files_list:
        try:
            img = tf.io.read_file(file)
            img1 = tf.image.decode_png(img, channels=3)
        except:
            os.remove(file)
            cnt_removed += 1
            print(file)

    print(f'DELETED: {cnt_removed}')


class ModularSupConLoss1(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, loss_mode="supcon", alpha=0.5, name="ModularSupConLoss"):
        super().__init__(name=name)
        self.temperature = temperature
        self.loss_mode = loss_mode.lower()  #"supcon", "logsum", "hybrid", "pairwise"
        self.alpha = tf.Variable(alpha, trainable=False, dtype=tf.float32)  #for hybrid

    def call(self, features, labels):
        #Normalize features
        #features = tf.math.l2_normalize(features, axis=-1)

        batch_size = tf.shape(features)[0]
        num_views = tf.shape(features)[1]

        #Flatten: [B, V, D] -> [B*V, D]
        features_flat = tf.reshape(features, [batch_size * num_views, -1])

        #Compute similarity matrix
        logits = tf.matmul(features_flat, features_flat, transpose_b=True)
        logits /= self.temperature

        #Create mask for positives
        labels = tf.argmax(labels, axis=-1)
        labels = tf.reshape(tf.tile(tf.expand_dims(labels, 1), [1, num_views]), [-1])
        mask = tf.cast(tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)), tf.float32)

        #Remove self-contrast cases
        logits_mask = tf.ones_like(mask) - tf.eye(batch_size * num_views)
        positives_mask = mask * logits_mask
        negatives_mask = logits_mask - positives_mask
        negatives_mask = tf.clip_by_value(negatives_mask, 0.0, 1.0)

        num_positives_per_row = tf.reduce_sum(positives_mask, axis=1)

        #--- SupCon (outside-log) ---
        if self.loss_mode == "supcon":
            logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
            exp_logits = tf.exp(logits)

            denom = tf.reduce_sum(exp_logits * logits_mask, axis=1, keepdims=True)
            log_probs = (logits - tf.math.log(denom + 1e-9)) * positives_mask
            log_probs = tf.reduce_sum(log_probs, axis=1)
            log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)

            loss = -log_probs
            loss = tf.reduce_mean(loss)

        #--- LogSum (inside-log) ---
        elif self.loss_mode == "logsum":
            logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
            exp_logits = tf.exp(logits)
            denom = tf.reduce_sum(exp_logits * logits_mask, axis=1, keepdims=True)
            numerator = tf.reduce_sum(exp_logits * positives_mask, axis=1, keepdims=True)

            probs = numerator / (denom + 1e-9)
            loss = -tf.math.log(probs + 1e-9)
            loss = tf.reduce_mean(loss)

        #--- Hybrid: weighted combination ---
        elif self.loss_mode == "hybrid":
            logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
            exp_logits = tf.exp(logits)
            denom = tf.reduce_sum(exp_logits * logits_mask, axis=1, keepdims=True)
            log_probs = (logits - tf.math.log(denom + 1e-9)) * positives_mask
            log_probs = tf.reduce_sum(log_probs, axis=1)
            log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)

            loss_out = -log_probs
            loss_out = tf.reduce_mean(loss_out)

            numerator = tf.reduce_sum(exp_logits * positives_mask, axis=1)
            probs_in = numerator / (denom[:, 0] + 1e-9)
            loss_in = -tf.math.log(probs_in + 1e-9)
            loss_in = tf.reduce_mean(loss_in)

            loss = self.alpha * loss_out + (1.0 - self.alpha) * loss_in

        #--- Pairwise-enhanced ---
        elif self.loss_mode == "pairwise":
            logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
            exp_logits = tf.exp(logits)

            def compute_pairwise_log(i):
                pos_indices = tf.where(positives_mask[i] > 0)[:, 0]
                sims = exp_logits[i]
                denom = tf.reduce_sum(sims * logits_mask[i])

                pair_terms = []
                num_pos = tf.shape(pos_indices)[0]
                for j in tf.range(num_pos):
                    for k in tf.range(j + 1, num_pos):
                        numerator = sims[pos_indices[j]] + sims[pos_indices[k]]
                        pair_terms.append(tf.math.log(numerator / (denom + 1e-9)))

                return -tf.reduce_mean(tf.stack(pair_terms)) if tf.shape(pair_terms)[0] > 0 else tf.constant(0.0)

            log_probs = tf.map_fn(compute_pairwise_log, tf.range(tf.shape(features_flat)[0]), dtype=tf.float32)
            loss = tf.reduce_mean(log_probs)

        else:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}")

        return loss

@enum.unique
class ModelMode(enum.Enum):
    TRAIN = 1
    EVAL = 2
    INFERENCE = 3


@enum.unique
class AugmentationType(enum.Enum):
    """Valid augmentation types."""
    #SimCLR augmentation (Chen et al, https://arxiv.org/abs/2002.05709).
    SIMCLR = 's'
    #AutoAugment augmentation (Cubuk et al, https://arxiv.org/abs/1805.09501).
    AUTOAUGMENT = 'a'
    #RandAugment augmentation (Cubuk et al, https://arxiv.org/abs/1909.13719).
    RANDAUGMENT = 'r'
    #SimCLR combined with RandAugment.
    STACKED_RANDAUGMENT = 'sr'
    #No augmentation.
    IDENTITY = 'i'


@enum.unique
class LossContrastMode(enum.Enum):
    ALL_VIEWS = 'a'  #All views are contrasted against all other views.
    ONE_VIEW = 'o'  #Only one view is contrasted against all other views.


@enum.unique
class LossSummationLocation(enum.Enum):
    OUTSIDE = 'o'  #Summation location is outside of logarithm
    INSIDE = 'i'  #Summation location is inside of logarithm


@enum.unique
class LossDenominatorMode(enum.Enum):
    ALL = 'a'  #All negatives and all positives
    ONE_POSITIVE = 'o'  #All negatives and one positive
    ONLY_NEGATIVES = 'n'  #Only negatives


@enum.unique
class Optimizer(enum.Enum):
    RMSPROP = 'r'
    MOMENTUM = 'm'
    LARS = 'l'
    ADAM = 'a'
    NESTEROV = 'n'


@enum.unique
class EncoderArchitecture(enum.Enum):
    RESNET_V1 = 'r1'
    RESNEXT = 'rx'


@enum.unique
class DecayType(enum.Enum):
    COSINE = 'c'
    EXPONENTIAL = 'e'
    PIECEWISE_LINEAR = 'p'
    NO_DECAY = 'n'


@enum.unique
class EvalCropMethod(enum.Enum):
    """Methods of cropping eval images to the target dimensions."""
    #Resize so that min image dimension is IMAGE_SIZE + CROP_PADDING, then crop
    #The central IMAGE_SIZExIMAGE_SIZE square.
    RESIZE_THEN_CROP = 'rc'
    #Crop a central square of side length
    #Natural_image_min_dim * IMAGE_SIZE/(IMAGE_SIZE+CROP_PADDING), then resize to
    #IMAGE_SIZExIMAGE_SIZE.
    CROP_THEN_RESIZE = 'cr'
    #Crop the central IMAGE_SIZE/(IMAGE_SIZE+CROP_PADDING) pixels along each
    #Dimension, preserving the natural image aspect ratio, then resize to
    #IMAGE_SIZExIMAGE_SIZE, which distorts the image.
    CROP_THEN_DISTORT = 'cd'
    #Do nothing. Requires that the input image is already the desired size.
    IDENTITY = 'i'


# Contrastive training wrapper around an encoder and projection head.
class Contrastive_Model(tf.keras.Model):
    """Model used for supervised contrastive pretraining and evaluation.

    This class owns three key pieces:
    1) the image augmenter for contrastive views,
    2) the encoder backbone, and
    3) the projection head used by contrastive losses.
    """

    def __init__(self, encoder):
        """Build the training graph components used during contrastive pretraining."""
        super().__init__()

        self.current_epoch = 0
        self.AUG_MIN_AREA = 0.8 #0.2#0.8
        self.AUG_ROTATION = 0.10 #0.40#0.10
        self.temperature = TEMPERATURE
        self.contrastive_augmenter = get_augmenter_echo(IMAGE_SIZE, self.AUG_MIN_AREA, self.AUG_ROTATION)#Contrastive_augmenter
        self.encoder = encoder
        self.num_classes = NUM_CLASSES#Num_classes

        self.supercon_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2])),
            tf.keras.layers.Rescaling(1. / 255.0),
            encoder,
            #Tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D()
        ])

        self.temp = self.supercon_model.output_shape
        self.sub_model_output_shape = (self.supercon_model.output_shape[1])

        #Non-linear MLP as projection head
        self.projection_head = tf.keras.Sequential(
        [
                tf.keras.Input(shape=self.sub_model_output_shape),  #Output shape of the encoder
                tf.keras.layers.Dense(NETWORK_WIDTH, activation="relu"),
                tf.keras.layers.Dense(NETWORK_WIDTH),
            ],
            name="Projection_Head",
        )

        #Self.con_loss = ModularSupConLoss(
        #temperature=TEMPERATURE,
        #loss_mode=USE_LOSS, #options: "supcon", "logsum", "hybrid", "pairwise",
        #alpha = 0.5 #for hybrid loss
        #)

        self.supercon_model.summary()
        self.projection_head.summary()
        self.stepcount = 0
        self.epoch_count = 0
        self.loss_variant = USE_LOSS#LOSSES[0]
        self.drop_k = DROP_K
        self.drop_threshold = DROP_V

    def compile(self, contrastive_optimizer, **kwargs):
        """Attach optimizer/metrics for the custom contrastive training loop."""
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer

        #Self.contrastive_loss will be defined as a method

        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        """Expose metric objects so Keras can reset/log them each epoch."""
        return [
            self.contrastive_loss_tracker,
        ]

    def _cap_positives_mask(self, untiled_mask, diagonal_mask, num_views, positives_cap):
        r"""Cap positives in the provided untiled_mask.

            'positives_cap' specifies the maximum number of positives *other* than
            augmentations of the anchor. Positives will be evenly sampled from all
            views.

        Args:
          untiled_mask: Tensor of shape [local_batch_size, global_batch_size] that has
            entry (r, c) == 1 if feature entries in rows r and c are from the same
            class. Else (r, c) == 0.
          diagonal_mask: Tensor with the same shape as `untiled_mask`. When
            local_batch_size == global_batch_size this is just an identity matrix.
            Otherwise, it is an identity matrix of size `local_batch_size` that is
            padded with 0's in the 2nd dimension to match the target shape. This is
            used to indicate where the anchor views exist in the global batch of
            views.
          num_views: Integer number of total views.
          positives_cap: Integer maximum number of positives *other* than
            augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
            Including augmentations, a maximum of (positives_cap + num_views - 1)
            positives is possible. This parameter modifies the contrastive numerator
            by selecting which positives are present in the summation, and which
            positives contribure to the denominator if denominator_mode ==
            enums.LossDenominatorMode.ALL.

        Returns:
          A tf.Tensor with the modified `untiled_mask`.
        """
        untiled_mask_no_diagonal = tf.math.minimum(untiled_mask, 1. - diagonal_mask)
        untiled_positives_per_anchor = positives_cap // num_views

        # Pick top-k candidates per anchor.
        # If a row has fewer than k true positives, temporary false positives may appear,
        # but they are removed before returning.
        _, top_k_col_idx = tf.math.top_k(untiled_mask_no_diagonal,
                                         untiled_positives_per_anchor)
        top_k_row_idx = tf.expand_dims(tf.range(tf.shape(untiled_mask)[0]), axis=1)

        if False:
            #Compute cosine similarities between anchor and all positives
            anchor_features = tf.reshape(self.last_anchor_features, [tf.shape(untiled_mask)[0], -1])  #[B, D]
            global_features = tf.reshape(self.last_all_features, [tf.shape(untiled_mask)[1], -1])  #[B, D]

            similarities = tf.linalg.matmul(anchor_features, global_features, transpose_b=True)  #[B, B]
            #Mask out non-positives for ranking
            similarities = similarities * untiled_mask_no_diagonal  #Mask non-positives

            #Get top-k most similar positives per row
            _, top_k_col_idx = tf.math.top_k(similarities, k=untiled_positives_per_anchor)
            top_k_row_idx = tf.expand_dims(tf.range(tf.shape(untiled_mask)[0]), axis=1)

        # Build scatter indices with shape [num_indices, 2] so we can mark selected
        # positives inside the tiled mask.
        top_k_idx = tf.reshape(
            tf.stack([
                tf.tile(top_k_row_idx,
                        (1, untiled_positives_per_anchor)), top_k_col_idx
            ],
                axis=-1), (-1, 2))

        # Scatter selected indices back into a capped untiled mask.
        untiled_mask_capped = tf.scatter_nd(
            top_k_idx,
            tf.ones(
                shape=tf.shape(top_k_idx)[0], dtype=untiled_mask_no_diagonal.dtype),
            untiled_mask_no_diagonal.shape)
        untiled_mask_capped = tf.math.maximum(untiled_mask_capped, diagonal_mask)
        return untiled_mask * untiled_mask_capped

    def _create_tiled_masks(self, untiled_mask, diagonal_mask, num_views,
                            num_anchor_views, positives_cap):
        r"""Creates tiled versions of untiled mask.

        Tiles `untiled_mask`, which has shape [local_batch_size, global_batch_size]
        by factors of [num_anchor_views, num_views], and then generates two masks from
        it. In both cases, the mask dimensions are ordered by view and then by sample,
        so if there was a batch size of 3 with 2 views the order would be
        [b1v1, b2v1, b3v1, b1v2, b2v2, b3v2]:
          positives_mask: Entry (row = i, col = j) is 1 if
            untiled_mask[i % local_batch_size, j % global_batch_size] == 1 and
            i // local_batch_size != j // global_batch_size. This results in a mask
            that is 1 for all pairs that are the same class but are not the exact same
            view. An exception to this is if positives_cap > -1, in which case there
            is a maximum of (positives_cap) 1-values per row, not including the
            entries that correspond to other views of the anchor. That is,
            positives_cap does nothing if there is only a single 1-valued entry per
            row in `untiled_mask`.
          negatives_mask: Entry (row = i, col = j) is 1 if features i and j are
            different classes. Otherwise the entry is 0.

        Args:
          untiled_mask: Tensor of shape [local_batch_size, global_batch_size], where
            local_batch_size <= global_batch_size, that has entry (r, c) == 1 if
            feature entries in rows r and c are from the same class. Else (r, c) == 0.
            In the self-supervised case, where the only positives are other views of
            the same sample, `untiled_mask` and `diagonal_mask` should be the same.
          diagonal_mask: Tensor with the same shape as `untiled_mask`. When
            local_batch_size == global_batch_size this is just an identity matrix.
            Otherwise, it is a slice of a [global_batch_size, global_batch_size]
            identity matrix that indicates where in the global batch the local batch
            is located.
          num_views: Integer number of total views.
          num_anchor_views: Integer number of anchor views.
          positives_cap: Integer maximum number of positives *other* than
            augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
            Including augmentations, a maximum of (positives_cap + num_views - 1)
            positives is possible. This parameter modifies the contrastive numerator
            by selecting which positives are present in the summation, and which
            positives contribure to the denominator if denominator_mode ==
            enums.LossDenominatorMode.ALL.

        Returns:
          Tuple containing positives_mask and negatives_mask tensors.
        """
        global_batch_size = tf.shape(untiled_mask)[1]
        # Build a mask that removes self-comparisons (same sample, same view).
        labels = tf.argmax(diagonal_mask, axis=-1)
        tiled_labels = []
        for i in range(num_anchor_views):
            tiled_labels.append(labels + tf.cast(global_batch_size, labels.dtype) * i)
        tiled_labels = tf.concat(tiled_labels, axis=0)
        tiled_diagonal_mask = tf.one_hot(tiled_labels, global_batch_size * num_views)
        all_but_diagonal_mask = 1. - tiled_diagonal_mask

        # Build uncapped positives and negatives on the tiled grid.
        uncapped_positives_mask = tf.tile(untiled_mask, [num_anchor_views, num_views])

        negatives_mask = 1. - uncapped_positives_mask

        # Optionally cap positives per anchor (excluding diagonal/self entries).
        if positives_cap > -1:
            untiled_mask = self._cap_positives_mask(untiled_mask, diagonal_mask, num_views,
                                               positives_cap)
            # Re-tile the capped untiled mask to get final positive entries.
            positives_mask = tf.tile(untiled_mask, [num_anchor_views, num_views])
        else:
            positives_mask = uncapped_positives_mask

        positives_mask = positives_mask * all_but_diagonal_mask  # Remove diagonal/self entries.

        return positives_mask, negatives_mask

    def _validate_contrastive_loss_inputs(self, features, labels, contrast_mode,
                                          summation_location, denominator_mode,
                                          positives_cap):
        r"""Validates inputs for contrastive_loss().

        Args:
          features: Tensor of rank at least 3, where the first 2 dimensions are
            batch_size and num_views, and the remaining dimensions are the feature
            shape.
          labels: One-hot labels tensor of shape [batch_size, num_labels] with numeric
            dtype.
          contrast_mode: LossContrastMode specifying which views get used as anchors.
          summation_location: LossSummationLocation specifying location of positives
            summation. See documentation above for more details.
          denominator_mode: LossDenominatorMode specifying which positives to include
            in contrastive denominator. See documentation above for more details.
          positives_cap: Integer maximum number of positives *other* than
            augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
            Including augmentations, a maximum of (positives_cap + num_views - 1)
            positives is possible. This parameter modifies the contrastive numerator
            by selecting which positives are present in the summation, and which
            positives contribure to the denominator if denominator_mode ==
            enums.LossDenominatorMode.ALL.

        Returns:
          Tuple containing batch_size and num_views values.

        Raises:
          ValueError if any of the inputs are invalid.
        """
        if features.shape.rank < 3:
            raise ValueError(
                f'Invalid features rank ( = {features.shape.rank}). Should have rank '
                '>= 3 with shape [batch_size, num_views] + `feature shape.`')

        #temp_shape = tf.shape(features)
        #batch_size = temp_shape[0]
        #batch_size = tf.compat.dimension_at_index(features.shape, 0).value
        batch_size = tf.shape(features)[0]
        batch_size = tf.cast(batch_size, tf.int32)
        if batch_size is None:
            raise ValueError('features has unknown batch_size dimension.')
        num_views = tf.compat.dimension_at_index(features.shape, 1).value
        #num_views = temp_shape[1]
        if num_views is None:
            raise ValueError('features has unknown num_views dimension.')

        #labels_shape = tf.shape(labels)
        #if labels is not None:
        #Check that |labels| are shaped like a one_hot vector.
        #if labels.shape.rank != 2 or labels.shape[0] != batch_size:
        #Raise ValueError(
        #F'Invalid labels shape (= {labels.shape}). Should have shape '
        #F'(batch_size = {batch_size}, num_labels).')

        if not isinstance(contrast_mode, LossContrastMode):
            raise ValueError(
                f'Invalid contrast_mode (= {contrast_mode}). Should be an instance of '
                'LossContrastMode.')
        if not isinstance(summation_location, LossSummationLocation):
            raise ValueError(
                f'Invalid summation_location (= {summation_location}). Should be an '
                'instance of LossSummationLocation.')
        if not isinstance(denominator_mode, LossDenominatorMode):
            raise ValueError(
                f'Invalid denominator_mode (= {denominator_mode}). Should be an '
                'instance of LossDenominatorMode.')
        if positives_cap > -1 and positives_cap % num_views != 0:
            raise ValueError(
                f'positives_cap (= {positives_cap}) must be a multiple of the '
                f'num_views (= {num_views}).')

        return batch_size, num_views
    

    @tf.function
    def true_negative_emphasis(self, sim, low=0.3, high=0.7, delta=0.1, k=30.0):
        """
        Emphasize true negatives (medium similarity).
        Similarity is assumed to be in [0, 1].

        Args:
            sim: tf.Tensor, similarity scores in [0, 1]
            low, high: bump window range (region of true negatives)
            delta: maximum logit boost for true negatives
            k: steepness of the sigmoid transitions

        Returns:
            Adjusted similarity tensor
        """
        sim = tf.convert_to_tensor(sim, dtype=tf.float32)

        left = 1 / (1 + tf.exp(-k * (sim - low)))
        right = 1 / (1 + tf.exp(-k * (sim - high)))
        bump = left * (1 - right)

        return sim + delta * bump  # Smoothly boost mid-range similarities only.

    def contrastive_loss(self, features,
                         labels=None,
                         temperature=1.0,
                         contrast_mode=LossContrastMode.ALL_VIEWS,
                         summation_location=LossSummationLocation.OUTSIDE,
                         denominator_mode=LossDenominatorMode.ALL,
                         positives_cap=-1,
                         scale_by_temperature=True,
                         loss_variant="supcon",
                         hybrid_alpha=0.5):
        """Compute per-sample contrastive loss for the selected loss variant.

        High-level flow:
        1) validate/reshape inputs,
        2) build logits and positive/negative masks,
        3) compute loss according to `loss_variant`,
        4) reduce to one loss value per sample.
        """
        # Step 1: standardize inputs so all loss variants see the same tensor shapes.
        features = tf.convert_to_tensor(features)
        labels = tf.convert_to_tensor(labels) if labels is not None else None

        local_batch_size, num_views = self._validate_contrastive_loss_inputs(
            features, labels, contrast_mode, summation_location, denominator_mode,
            positives_cap)

        if features.shape.rank > 3:
            features = tf.reshape(features, tf.concat([tf.shape(features)[:2], [-1]], axis=0))
        if features.dtype != tf.float32:
            features = tf.cast(features, tf.float32)

        # Step 2: prepare global feature and mask context for pairwise comparisons.
        global_features = features
        global_batch_size = tf.shape(global_features)[0]
        global_batch_size = tf.cast(global_batch_size, tf.int32)
        local_replica_id = 0

        diagonal_mask = tf.one_hot(tf.range(local_batch_size) + (local_replica_id * local_batch_size), global_batch_size)

        if labels is None:
            mask = diagonal_mask
        else:
            labels = tf.cast(labels, tf.float32)
            global_labels = labels
            mask = tf.linalg.matmul(labels, global_labels, transpose_b=True)
        tf.debugging.assert_equal(tf.shape(mask), [local_batch_size, global_batch_size])

        all_global_features = tf.reshape(tf.transpose(global_features, perm=[1, 0, 2]), [num_views * global_batch_size, -1])

        if contrast_mode == LossContrastMode.ONE_VIEW:
            anchor_features = features[:, 0]
            num_anchor_views = 1
        else:
            anchor_features = tf.reshape(tf.transpose(features, perm=[1, 0, 2]), [num_views * local_batch_size, -1])
            num_anchor_views = num_views

        # Step 3: compute logits and exponentiated logits used by all variants.
        raw_logits = tf.linalg.matmul(anchor_features, all_global_features, transpose_b=True)
        temperature = tf.cast(temperature, tf.float32)
        logits = raw_logits / temperature
        logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
        exp_logits = tf.exp(logits)

        positives_mask, negatives_mask = (
            self._create_tiled_masks(mask, diagonal_mask, num_views, num_anchor_views,
                                     positives_cap))
        num_positives_per_row = tf.reduce_sum(positives_mask, axis=1)

        # Step 4: choose denominator behavior (all positives, one positive, or negatives-only).
        if denominator_mode == LossDenominatorMode.ALL:
            denominator = tf.reduce_sum(exp_logits * negatives_mask, axis=1, keepdims=True) + tf.reduce_sum(exp_logits * positives_mask, axis=1, keepdims=True)
        elif denominator_mode == LossDenominatorMode.ONE_POSITIVE:
            denominator = exp_logits + tf.reduce_sum(exp_logits * negatives_mask, axis=1, keepdims=True)
        else:
            denominator = tf.reduce_sum(exp_logits * negatives_mask, axis=1, keepdims=True)

        # Step 5: dispatch to the selected objective variant.
        if loss_variant == "supcon":
            log_probs = (logits - tf.math.log(denominator)) * positives_mask
            log_probs = tf.reduce_sum(log_probs, axis=1)
            log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
            loss = -log_probs

        elif loss_variant == "dropcon":

            k = self.drop_k
            v = self.drop_threshold  #New: minimum number of positives required to apply DropCon

            #Compute per-positive log-probs
            log_probs = (logits - tf.math.log(denominator)) * positives_mask  #[N, N]

            #Get raw similarities for positives only
            sims = raw_logits * positives_mask  #[N, N]
            sims = tf.where(positives_mask > 0, sims, tf.fill(tf.shape(sims), -1.0))  #Replace non-positives with -1

            #Sort similarities in ascending order
            sorted_sims = tf.sort(sims, direction='ASCENDING', axis=-1)  #[N, N]

            #Get the k-th smallest similarity (drop threshold)
            kth_sim = tf.gather(sorted_sims, k, axis=1)  #[N]
            kth_sim_expanded = tf.expand_dims(kth_sim, axis=1)  #[N, 1]

            #Build keep mask: keep positives above the kth similarity
            keep_mask = tf.cast(sims > kth_sim_expanded, tf.float32) * positives_mask  #[N, N]

            #Count valid positives and kept ones
            num_valid_pos = tf.reduce_sum(positives_mask, axis=1)  #[N]
            num_kept = tf.reduce_sum(keep_mask, axis=1)  #[N]

            #Fallback condition: only apply DropCon if num positives > v
            apply_dropcon_mask = tf.cast(num_valid_pos > v, tf.float32)  #[N]
            fallback_mask = 1.0 - apply_dropcon_mask

            #DropCon loss
            masked_log_probs = log_probs * keep_mask
            dropcon_log_probs = tf.reduce_sum(masked_log_probs, axis=1)
            dropcon_log_probs = tf.math.divide_no_nan(dropcon_log_probs, num_kept)  #[N]

            #Standard SupCon fallback
            fallback_log_probs = tf.reduce_sum(log_probs, axis=1)
            fallback_log_probs = tf.math.divide_no_nan(fallback_log_probs, num_valid_pos)  #[N]

            #Final combined loss
            loss = - (apply_dropcon_mask * dropcon_log_probs + fallback_mask * fallback_log_probs)

        #elif loss_variant == "compcon": #either adds or replaces, not both
        #augment_only = True #Set to False to use "replace-weak" mode
        #alpha = 1.0 #Controls how far below the mean a positive must be to be dropped
        #eps = 1e-9
        #Similarity matrix (assumes normalized features, so dot product is cosine sim)
        #sims = raw_logits #[N, N]
        #Get masks
        #pos_mask = positives_mask #[N, N]
        #neg_mask = negatives_mask #[N, N]
        #Compute mean and std of positive similarities per anchor
        #masked_pos_sims = tf.where(pos_mask > 0, sims, tf.zeros_like(sims)) #[N, N]
        #num_pos = tf.reduce_sum(pos_mask, axis=1, keepdims=True) #[N, 1]
        #mean_pos = tf.reduce_sum(masked_pos_sims, axis=1, keepdims=True) / (num_pos + eps) #[N, 1]
        #Compute variance and std
        #sq_diff = tf.square(masked_pos_sims - mean_pos)
        #var_pos = tf.reduce_sum(sq_diff, axis=1, keepdims=True) / (num_pos + eps)
        #std_pos = tf.sqrt(var_pos + eps) #[N, 1]
        #Adaptive threshold: mean - alpha * std
        #adaptive_thresh = mean_pos - alpha * std_pos #[N, 1]
        #thresh_matrix = tf.tile(adaptive_thresh, [1, tf.shape(sims)[1]]) #[N, N]
        #Candidates: negatives with higher sim than the adaptive threshold
        #candidate_mask = tf.cast(sims > thresh_matrix, tf.float32) * neg_mask #[N, N]
        #if augment_only:
        #Augment positives with strong negatives
        #final_pos_mask = pos_mask + candidate_mask #[N, N]
        #else:
        #Replace weak positives (sim below threshold) with strong negatives
        #strong_pos_mask = tf.cast(sims >= thresh_matrix, tf.float32) * pos_mask
        #final_pos_mask = strong_pos_mask + candidate_mask #[N, N]
        #Compute logits
        #logits = raw_logits / temperature
        #logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
        #exp_logits = tf.exp(logits)
        #Compute log-probabilities
        #denominator = tf.reduce_sum(exp_logits * (pos_mask + neg_mask), axis=1, keepdims=True)
        #log_probs = (logits - tf.math.log(denominator + eps)) * final_pos_mask
        #Reduce
        #summed_log_probs = tf.reduce_sum(log_probs, axis=1)
        #num_final_pos = tf.reduce_sum(final_pos_mask, axis=1)
        #avg_log_probs = tf.math.divide_no_nan(summed_log_probs, num_final_pos)
        #Final loss
        #loss = -avg_log_probs

        elif loss_variant == "compcon_hybrid": #Automatically adds or removes(not replace, actually drops it) based on mean and std
            alpha = 1.1  #Threshold for weak positives
            beta = 0.05  #Margin to decide whether to replace
            eps = 1e-9

            sims = raw_logits  #Cosine similarity, assumed in [0,1]

            pos_mask = positives_mask  #[N, N]
            neg_mask = negatives_mask  #[N, N]

            #Compute stats over positives
            masked_pos_sims = tf.where(pos_mask > 0, sims, tf.zeros_like(sims))
            num_pos = tf.reduce_sum(pos_mask, axis=1, keepdims=True)
            mean_pos = tf.reduce_sum(masked_pos_sims, axis=1, keepdims=True) / (num_pos + eps)

            sq_diff = tf.square(masked_pos_sims - mean_pos)
            var_pos = tf.reduce_sum(sq_diff, axis=1, keepdims=True) / (num_pos + eps)
            std_pos = tf.sqrt(var_pos + eps)

            thresh = mean_pos - alpha * std_pos  #Adaptive threshold [N, 1]
            margin_thresh = mean_pos + beta  #Replacement margin [N, 1]

            #Broadcast
            thresh_matrix = tf.tile(thresh, [1, tf.shape(sims)[1]])
            margin_thresh_matrix = tf.tile(margin_thresh, [1, tf.shape(sims)[1]])

            #Determine weak positives and strong negatives
            weak_pos_mask = tf.cast(sims < thresh_matrix, tf.float32) * pos_mask
            strong_pos_mask = tf.cast(sims >= thresh_matrix, tf.float32) * pos_mask
            strong_neg_mask = tf.cast(sims > margin_thresh_matrix, tf.float32) * neg_mask

            #Final hybrid mask
            final_pos_mask = strong_pos_mask + strong_neg_mask

            #Compute logits
            logits = raw_logits / temperature
            logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
            exp_logits = tf.exp(logits)

            #Log-probs
            denominator = tf.reduce_sum(exp_logits * (pos_mask + neg_mask), axis=1, keepdims=True)
            log_probs = (logits - tf.math.log(denominator + eps)) * final_pos_mask

            #Reduce
            summed_log_probs = tf.reduce_sum(log_probs, axis=1)
            num_final_pos = tf.reduce_sum(final_pos_mask, axis=1)
            avg_log_probs = tf.math.divide_no_nan(summed_log_probs, num_final_pos)

            #Fallback mask: if no final positives, use standard SupCon
            fallback_mask = tf.cast(num_final_pos <= 0, tf.float32)  #[N]

            #Standard SupCon log-probs
            log_probs_std = (logits - tf.math.log(denominator + eps)) * pos_mask
            summed_std = tf.reduce_sum(log_probs_std, axis=1)
            num_std = tf.reduce_sum(pos_mask, axis=1)
            avg_std = tf.math.divide_no_nan(summed_std, num_std)

            #Combine: use hybrid unless fallback triggered
            loss = -((1. - fallback_mask) * avg_log_probs + fallback_mask * avg_std)

        elif loss_variant == "compcon": #Uses the lowest sim positive instead of the mean or std
            augment_only = True  #Set to False to use "replace-weak" mode
            delta = 0.02
            eps = 1e-9

            #Similarity matrix (assumes normalized features, so dot product is cosine sim)
            sims = raw_logits  #[N, N]

            #Get masks
            pos_mask = positives_mask  #[N, N]
            neg_mask = negatives_mask  #[N, N]

            #Minimum positive similarity per anchor (masked)
            min_pos_sim = tf.reduce_min(tf.where(pos_mask > 0, sims, tf.ones_like(sims) * 1e9), axis=1,
                                        keepdims=True)  #[N, 1]
            min_pos_sim_exp = tf.tile(min_pos_sim + delta, [1, tf.shape(sims)[1]])  #[N, N]

            #Identify strong negatives: those higher than the weakest positive + margin
            candidate_mask = tf.cast((sims > min_pos_sim_exp), tf.float32) * neg_mask  #[N, N]

            if augment_only:
                #Augment positives with strong negatives
                final_pos_mask = pos_mask + candidate_mask  #[N, N]
            else:
                #Replace weakest positives with stronger negatives
                keep_pos_mask = tf.cast(sims >= min_pos_sim_exp, tf.float32) * pos_mask  #Keep strong positives
                final_pos_mask = keep_pos_mask + candidate_mask  #[N, N]

            #Compute logits
            logits = raw_logits / temperature
            logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
            exp_logits = tf.exp(logits)

            #Compute denominators and log-probs
            log_probs = (logits - tf.math.log(denominator)) * final_pos_mask  #[N, N]

            #Average across final positives
            summed_log_probs = tf.reduce_sum(log_probs, axis=1)
            num_final_pos = tf.reduce_sum(final_pos_mask, axis=1)
            avg_log_probs = tf.math.divide_no_nan(summed_log_probs, num_final_pos)

            #Final loss
            loss = -avg_log_probs

        elif loss_variant == "supcon_softscale":

            #Raw cosine similarities
            #sims = raw_logits * positives_mask
            #sims = tf.where(positives_mask > 0, sims, tf.zeros_like(sims))
            #Weight with soft power
            #gamma = 0.5 #Tune between 0.3 and 0.9
            #weights = tf.pow(sims + 1e-6, gamma) #ensure differentiable and avoid zero
            #log_probs = (logits - tf.math.log(denominator)) * positives_mask
            #log_probs = weights * log_probs
            #log_probs = tf.reduce_sum(log_probs, axis=1)
            #log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
            #loss = -log_probs


            #Log-probs like in SupCon
            log_probs = (logits - tf.math.log(denominator + 1e-9)) * positives_mask  #[N, N]

            #Get cosine similarities for positives only
            sims = raw_logits * positives_mask  #[N, N]

            #Fallback mask for anchors with <=1 positive
            fallback_mask = tf.cast(num_positives_per_row <= 1, tf.float32)

            #Stabilize similarity values (avoid 0^gamma or NaN)
            eps = 1e-6
            sims_clamped = tf.clip_by_value(sims, eps, 1.0)  #[N, N]

            #Soft scaling with exponent gamma (tunable, e.g., gamma = 1.5)
            gamma = 0.5
            weights = tf.pow(sims_clamped, gamma) * positives_mask  #[N, N]

            #Apply weights to log-probs
            weighted_log_probs = tf.reduce_sum(weights * log_probs, axis=1)  #[N]

            #Fallback: standard SupCon where there's only 1 positive
            standard_log_probs = tf.reduce_sum(log_probs, axis=1)
            standard_log_probs = tf.math.divide_no_nan(standard_log_probs, num_positives_per_row)
            fallback_loss = -standard_log_probs  #[N]

            #Final loss: fallback if only 1 positive
            loss = -weighted_log_probs * (1. - fallback_mask) + fallback_loss * fallback_mask

        elif loss_variant == "supcon_sqmax":
            #Compute per-positive log-probs
            log_probs = (logits - tf.math.log(denominator + 1e-6)) * positives_mask  #[N, N]

            #Fallback mask: anchors with only 1 positive
            fallback_mask = tf.cast(num_positives_per_row <= 1, tf.float32)

            #Only use valid similarities
            sims = raw_logits * positives_mask  #[N, N]
            sims = tf.where(positives_mask > 0, sims, tf.zeros_like(sims))

            #Squared similarity weights
            w_sq = tf.square(sims)  #[N, N]

            #Max similarity per anchor (over valid positives only)
            max_sim = tf.reduce_max(tf.where(positives_mask > 0, sims, tf.zeros_like(sims)), axis=1, keepdims=True)
            max_sim_safe = tf.where(max_sim > 1e-6, max_sim, tf.ones_like(max_sim))  #Prevent div-by-zero

            #Max-normalized weights (unnormalized, no sum-to-one)
            w_max = tf.math.divide_no_nan(sims, max_sim_safe)  #[N, N]

            #Combine weights: geometric mean with stability and positivity clamp
            combined_weights = tf.sqrt(tf.nn.relu(w_sq * w_max) + 1e-6)  #[N, N]

            #Apply combined weights directly (no normalization)
            weighted_log_probs = tf.reduce_sum(combined_weights * log_probs, axis=1)  #[N]

            #Fallback to standard SupCon for anchors with only 1 positive
            standard_log_probs = tf.reduce_sum(log_probs, axis=1)
            standard_log_probs = tf.math.divide_no_nan(standard_log_probs, num_positives_per_row)
            fallback_loss = -standard_log_probs  #[N]

            #Final loss: use weighted where >= 2 positives, fallback otherwise
            loss = -weighted_log_probs * (1. - fallback_mask) + fallback_loss * fallback_mask

        elif loss_variant == "supcon_weighted": #Not good
            #Compute the per-positive log-probs
            log_probs = (logits - tf.math.log(denominator)) * positives_mask  #[N, N]

            #Use exp(logits) as weights, normalized over the positives
            weights = exp_logits * positives_mask  #[N, N]
            weights_sum = tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-9
            weights = weights / weights_sum  #Normalize weights over positives

            #Apply weights to log-probs and sum
            weighted_log_probs = tf.reduce_sum(weights * log_probs, axis=1)  #[N]

            #Final loss
            loss = -weighted_log_probs

        elif loss_variant == "logsum":
            numerator = exp_logits * positives_mask
            numerator_sum = tf.reduce_sum(numerator, axis=1)
            log_probs = tf.math.log(tf.math.divide_no_nan(numerator_sum, tf.squeeze(denominator, axis=1)))
            loss = -log_probs

        elif loss_variant == "logsum_weighted":
            #Weighted LogSum: Normalize positive logits across each row to use as weights
            weighted_numerator = exp_logits * positives_mask  #[N, N]
            numerator_sum = tf.reduce_sum(weighted_numerator, axis=1, keepdims=True) + 1e-9
            weights = weighted_numerator / numerator_sum  #[N, N], softmax-like weights

            #Compute weighted positive mass
            weighted_mass = tf.reduce_sum(weighted_numerator * weights, axis=1)  #[N]

            #Final logsum loss
            log_probs = tf.math.log(tf.math.divide_no_nan(weighted_mass, tf.squeeze(denominator, axis=1)))
            loss = -log_probs

        elif loss_variant == "logsum_neg_emph":
            #Apply bump emphasis to cosine similarities  in  [0, 1]
            emphasized_neg_sims   = self.true_negative_emphasis(raw_logits*negatives_mask, low=0.3, high=0.7, delta=0.15, k=30.0)
            #Combine original positive sims with emphasized negative sims
            emphasized_sims = (raw_logits * positives_mask) + emphasized_neg_sims

            #Compute adjusted logits and exp
            emphasized_logits = emphasized_sims / temperature
            emphasized_logits -= tf.reduce_max(tf.stop_gradient(emphasized_logits), axis=1, keepdims=True)
            exp_emphasized_logits = tf.exp(emphasized_logits)

            #Numerator remains original (clean positives only)
            numerator = exp_logits * positives_mask
            numerator_sum = tf.reduce_sum(numerator, axis=1)

            #Denominator includes both positive and emphasized negative terms
            denom = tf.reduce_sum(exp_emphasized_logits * negatives_mask, axis=1, keepdims=True) + tf.reduce_sum(
                exp_emphasized_logits * positives_mask, axis=1, keepdims=True)

            log_probs = tf.math.log(tf.math.divide_no_nan(numerator_sum, tf.squeeze(denom, axis=1)))
            loss = -log_probs

        elif loss_variant == "logsum_expavgsum":
            #Compute dot products for positives and sum them
            dot_sim = logits * positives_mask  #[N, N], only pos sims remain

            #Sum and average across positives for each anchor
            pos_sum = tf.reduce_sum(dot_sim, axis=1)  #[N]
            avg_pos_sim = tf.math.divide_no_nan(pos_sum, num_positives_per_row)  #[N]

            #Exponentiate the averaged similarity (numerator)
            numerator = tf.exp(avg_pos_sim / temperature)  #[N]

            #Denominator: same as denominator_mode==ALL
            denom = tf.reduce_sum(exp_logits * negatives_mask, axis=1) + tf.reduce_sum(
                exp_logits * positives_mask, axis=1)  #[N]

            #Final log-softmax loss
            log_probs = tf.math.log(tf.math.divide_no_nan(numerator, denom))  #[N]
            loss = -log_probs  #[N]

        elif loss_variant == "hybrid":
            log_probs_out = (logits - tf.math.log(denominator)) * positives_mask
            log_probs_out = tf.reduce_sum(log_probs_out, axis=1)
            log_probs_out = tf.math.divide_no_nan(log_probs_out, num_positives_per_row)
            loss_out = -log_probs_out

            log_probs_in = exp_logits * positives_mask
            log_probs_in = tf.reduce_sum(log_probs_in, axis=1)
            log_probs_in = tf.math.log(tf.math.divide_no_nan(log_probs_in, tf.squeeze(denominator, axis=1)))
            loss_in = -log_probs_in

            loss = hybrid_alpha * loss_out + (1.0 - hybrid_alpha) * loss_in

        #elif loss_variant == "pairwise":
        #logits_mask = 1. - tf.eye(tf.shape(anchor_features)[0])
        #def compute_pairwise_log(i):
        #pos_indices = tf.where(positives_mask[i] > 0)[:, 0]
        #sims = exp_logits[i]
        #denom_i = tf.reduce_sum(sims * logits_mask[i])
        #pair_terms = []
        #num_pos = tf.shape(pos_indices)[0]
        #for j in tf.range(num_pos):
        #for k in tf.range(j + 1, num_pos):
        #numerator = sims[pos_indices[j]] + sims[pos_indices[k]]
        #Pair_terms.append(tf.math.log(numerator / (denom_i + 1e-9)))
        #return -tf.reduce_mean(tf.stack(pair_terms)) if tf.shape(pair_terms)[0] > 0 else tf.constant(0.0)
        #log_probs = tf.map_fn(compute_pairwise_log, tf.range(tf.shape(anchor_features)[0]), dtype=tf.float32)
        #loss = log_probs

        elif loss_variant == "pairwise":

            eps = 1e-9
            N = tf.shape(anchor_features)[0]

            #Mask to exclude self-similarities from denominator
            logits_mask = 1. - tf.eye(N)

            #Similarities = exp(dot products)
            sims = exp_logits

            #Denominator for each anchor (exclude self)
            denom = tf.reduce_sum(sims * logits_mask, axis=1, keepdims=True)  #[N, 1]

            #Get all (anchor, positive) index pairs
            pos_mask = positives_mask > 0  #[N, N]
            pos_indices = tf.where(pos_mask)  #[num_pos, 2]

            #Gather anchor and positive indices
            anchor_ids = pos_indices[:, 0]
            pos_ids = pos_indices[:, 1]

            #Gather exp(sim(i, p)) values
            sim_values = tf.gather_nd(sims, pos_indices)  #[num_pos]

            #Group similarity values by anchor
            unique_ids, idx, counts = tf.unique_with_counts(anchor_ids)
            max_pos = tf.reduce_max(counts)  #Max number of positives for any anchor
            num_anchors = tf.shape(unique_ids)[0]

            #Convert to dense matrix: [num_anchors, max_pos]
            sim_values_padded = tf.RaggedTensor.from_value_rowids(sim_values, idx).to_tensor(
                shape=[num_anchors, max_pos], default_value=0.0)

            #Compute all pairwise sums of positive similarities per anchor
            sims_j = tf.expand_dims(sim_values_padded, axis=2)  #[B, P, 1]
            sims_k = tf.expand_dims(sim_values_padded, axis=1)  #[B, 1, P]
            pairwise_sums = sims_j + sims_k  #[B, P, P]

            #Keep only (j < k) upper triangle entries (no double-counting or self-pairs)
            triu_mask = tf.linalg.band_part(tf.ones_like(pairwise_sums), 0, -1) - tf.linalg.band_part(
                tf.ones_like(pairwise_sums), 0, 0)
            valid_pairs = pairwise_sums * triu_mask  #[B, P, P]

            #Avoid NaNs by replacing zeros with 1.0 (they will be masked out later)
            safe_mask = valid_pairs > 0
            safe_pairs = tf.where(safe_mask, valid_pairs, tf.ones_like(valid_pairs))

            #Denominator per anchor (repeat per anchor index)
            anchor_denom = tf.gather(denom, unique_ids)  #[B, 1]

            #Compute log( (sim_j + sim_k) / denom )
            log_terms = tf.math.log(tf.math.divide_no_nan(safe_pairs, tf.expand_dims(anchor_denom, axis=1)))
            log_terms = tf.where(safe_mask, log_terms, tf.zeros_like(log_terms))  #Zero out invalid terms

            #Average pairwise log-terms for each anchor
            pairwise_loss = -tf.reduce_mean(log_terms, axis=[1, 2])  #[B]

            #Compute SupCon fallback loss for anchors with only 1 positive
            full_logits = logits  #[N, N]
            supcon_denom = tf.reduce_sum(sims * logits_mask, axis=1, keepdims=True)
            supcon_log_probs = (full_logits - tf.math.log(supcon_denom + eps)) * positives_mask
            supcon_log_probs = tf.reduce_sum(supcon_log_probs, axis=1)
            num_pos_per_row = tf.reduce_sum(positives_mask, axis=1)
            supcon_log_probs = tf.math.divide_no_nan(supcon_log_probs, num_pos_per_row)
            supcon_loss = -supcon_log_probs  #[N]

            #Identify which anchors have >= 2 positives
            mask_pairwise = counts > 1  #[B] boolean

            #Scatter pairwise loss to full [N] vector
            pairwise_loss_full = tf.scatter_nd(
                indices=tf.expand_dims(unique_ids, 1),
                updates=tf.where(mask_pairwise, pairwise_loss, tf.zeros_like(pairwise_loss)),
                shape=[N],
            )

            #Create a mask of which anchors use pairwise
            mask_pairwise_full = tf.scatter_nd(
                indices=tf.expand_dims(unique_ids, 1), updates=tf.cast(mask_pairwise, tf.float32), shape=[N],)

            #Final per-anchor loss: pairwise if >=2 positives, SupCon otherwise
            loss = mask_pairwise_full * pairwise_loss_full + (1. - mask_pairwise_full) * supcon_loss

        elif loss_variant == "dcl_softmax":
            eps = 1e-9
            N = tf.shape(anchor_features)[0]

            #Compute pairwise similarities: [N, N]
            sims = tf.linalg.matmul(anchor_features, all_global_features, transpose_b=True)
            sims /= temperature

            #Mask for self-similarities
            logits_mask = 1. - tf.eye(N)

            #Apply mask to avoid self-comparisons
            sims = sims * logits_mask

            #Positive and negative masks
            pos_mask = tf.cast(positives_mask, tf.float32)
            pos_mask = tf.linalg.set_diag(pos_mask, tf.zeros([N]))  #Remove diagonal

            neg_mask = tf.cast(negatives_mask, tf.float32)
            neg_mask = tf.linalg.set_diag(neg_mask, tf.zeros([N]))

            #Average positive and negative similarities per anchor
            mu_pos = tf.math.divide_no_nan(tf.reduce_sum(sims * pos_mask, axis=1),
                                           tf.reduce_sum(pos_mask, axis=1) + eps)
            mu_neg = tf.math.divide_no_nan(tf.reduce_sum(sims * neg_mask, axis=1),
                                           tf.reduce_sum(neg_mask, axis=1) + eps)

            #Compute softmax contrastive loss
            exp_pos = tf.exp(mu_pos)
            exp_neg = tf.exp(mu_neg)
            denom = exp_pos + exp_neg
            log_softmax = tf.math.log(tf.math.divide_no_nan(exp_pos, denom + eps))
            loss = -log_softmax

        elif loss_variant == "sup_barlow":
            eps = 1e-9
            N = tf.shape(anchor_features)[0]

            #[N, D] - assume anchor_features are normalized if needed
            z = anchor_features  #Can also average across views if required

            #Use the label mask to define positive pairs (same class)
            label_mask = tf.cast(positives_mask, tf.float32)  #[N, N]
            label_mask = tf.linalg.set_diag(label_mask, tf.zeros([N]))  #Remove self-pairs

            #Normalize features across batch (zero mean, unit variance)
            z_mean = tf.reduce_mean(z, axis=0, keepdims=True)
            z_std = tf.math.reduce_std(z, axis=0, keepdims=True) + eps
            z_norm = (z - z_mean) / z_std

            #Compute supervised cross-correlation matrix
            norm_factor = tf.reduce_sum(label_mask) + eps
            C = tf.matmul(tf.transpose(z_norm), tf.matmul(label_mask, z_norm)) / norm_factor  #[D, D]

            #Diagonal loss (alignment): force to be close to 1
            on_diag = tf.reduce_sum(tf.square(tf.linalg.diag_part(C) - 1))

            #Off-diagonal loss (decorrelation): force to be close to 0
            off_diag = tf.reduce_sum(tf.square(C)) - tf.reduce_sum(tf.square(tf.linalg.diag_part(C)))

            lambda_offdiag = 0.005  #You can tune this
            loss = on_diag + lambda_offdiag * off_diag

        elif loss_variant == "supcon_adaptive":
            #Similarities before temperature scaling: [N, N]
            raw_logits = tf.matmul(anchor_features, all_global_features, transpose_b=True)

            #Compute std deviation across each anchor's row (excluding self-similarities)
            N = tf.shape(raw_logits)[0]
            logits_mask = 1. - tf.eye(N)
            sims_masked = raw_logits * logits_mask
            mean_sim = tf.reduce_sum(sims_masked, axis=1) / (tf.reduce_sum(logits_mask, axis=1) + 1e-9)
            var_sim = tf.reduce_sum(tf.square(sims_masked - tf.expand_dims(mean_sim, 1)) * logits_mask, axis=1) / (
                    tf.reduce_sum(logits_mask, axis=1) + 1e-9)
            std_sim = tf.sqrt(var_sim + 1e-9)

            #Adaptive temperature: lower std -> higher tau, higher std -> lower tau
            beta = 0.07  #Tunable parameter
            tau_i = tf.stop_gradient(beta / (std_sim + 1e-6))  #Shape [N]
            tau_i = tf.clip_by_value(tau_i, clip_value_min=0.03, clip_value_max=0.5)

            #Expand to match logits shape for elementwise division
            tau_matrix = tf.expand_dims(tau_i, axis=1)  #[N, 1]

            #Scale logits by adaptive temperature
            logits = raw_logits / tau_matrix
            logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
            exp_logits = tf.exp(logits)

            #Compute denominator
            if denominator_mode == LossDenominatorMode.ALL:
                denominator = tf.reduce_sum(exp_logits * negatives_mask, axis=1, keepdims=True) + tf.reduce_sum(
                    exp_logits * positives_mask, axis=1, keepdims=True)
            elif denominator_mode == LossDenominatorMode.ONE_POSITIVE:
                denominator = exp_logits + tf.reduce_sum(exp_logits * negatives_mask, axis=1, keepdims=True)
            else:
                denominator = tf.reduce_sum(exp_logits * negatives_mask, axis=1, keepdims=True)

            #SupCon log-probs
            log_probs = (logits - tf.math.log(denominator)) * positives_mask
            log_probs = tf.reduce_sum(log_probs, axis=1)
            log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
            loss = -log_probs

        else:
            raise ValueError(f"Unsupported loss_variant: {loss_variant}")

        # Step 6: apply final scaling/reduction to obtain one loss per sample.
        if loss_variant not in ['sup_barlow']:
            if scale_by_temperature:
                loss *= temperature
            loss = tf.reshape(loss, [num_anchor_views, local_batch_size])

            if num_views != 1:
                loss = tf.reduce_mean(loss, axis=0)
            else:
                num_valid_views_per_sample = tf.reshape(num_positives_per_row, [1, local_batch_size])
                loss = tf.squeeze(tf.math.divide_no_nan(loss, num_valid_views_per_sample))

        return loss


    def train_step(self, data):
        """Run one training step: augment -> encode -> project -> loss -> optimize."""
        self.stepcount += 1

        # Keras passes (images, labels). Keep names explicit for readability.
        images = data[0]
        labels = data[1]

        # Build one augmented view; the original image acts as the second view.

        augmented_images = self.contrastive_augmenter(images, training=False)

        with tf.GradientTape() as tape:
            features_1 = self.supercon_model(data[0], training=True)
            features_2 = self.supercon_model(augmented_images, training=True)

            # Pass both views through the projection head before contrastive loss.
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)

            projections_1 = tf.math.l2_normalize(projections_1, axis=1)
            projections_2 = tf.math.l2_normalize(projections_2, axis=1)

            labels = tf.one_hot(labels, self.num_classes)

            contrastive_loss = self.contrastive_loss(
                tf.stack([projections_1, projections_2], axis=1),
                labels=labels,
                temperature=self.temperature,
                loss_variant=self.loss_variant)

        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )

        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Run one validation step using the same view construction as training."""
        # Keras passes (images, labels). Keep names explicit for readability.
        images = data[0]
        labels = data[1]

        augmented_images = self.contrastive_augmenter(images, training=False)
        features_1 = self.supercon_model(data[0], training=False)
        features_2 = self.supercon_model(augmented_images, training=False)

        # Pass both views through the projection head before contrastive loss.
        projections_1 = self.projection_head(features_1, training=False)
        projections_2 = self.projection_head(features_2, training=False)

        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        labels = tf.one_hot(labels, self.num_classes)

        contrastive_loss = self.contrastive_loss(
            tf.stack([projections_1, projections_2], axis=1),
            labels=labels,
            temperature=self.temperature,
            loss_variant=self.loss_variant)

        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    def save_weights(self, filepath, overwrite=True):
        """Save encoder-only weights (used for downstream transfer)."""
        print(f'\n **SAVED WEIGHTS: f{filepath} \n')
        self.encoder.save_weights(filepath, overwrite)

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """Save the full contrastive model graph (encoder + projection pipeline)."""
        print(f'\n **SAVED MODEL: f{filepath} \n')
        self.supercon_model.save(filepath, overwrite, save_format)

class SelectiveModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, output_folder):
        super().__init__()
        self.output_folder = output_folder

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1  #Keras uses 0-based indexing
        if current_epoch == 5 or current_epoch == 15 or current_epoch % 10 == 0:
            save_path = os.path.join(self.output_folder, f"model_ep_{current_epoch}.h5")
            self.model.save(save_path)
            print(f"\n **SAVED MODEL: {save_path} \n ")

def  ssl_training(train_files, train_labels, val_files, val_labels, output_folder):
    #TODO: should I use val splitkappa

    print(f'Number of training files: {len(train_files)}')

    #Get tf dataset for self-supervised learning
    #ssl_val_perc = 0.025
    num_train = len(train_files)
    #num_val = int(num_train * ssl_val_perc)
    #val_files = train_files[:num_val]
    #train_files1 = train_files[num_val:]

    gpus = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5"]
    ssl_batch_size = SSL_BATCH_SIZE
    #ssl_batch_size = len(gpus) * SSL_BATCH_SIZE
    #Create a MirroredStrategy:
    #strategy = tf.distribute.MirroredStrategy(devices=gpus)
    #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    #print("-------------------------")
    #with strategy.scope():

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.map(process_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=SSL_BATCH_SIZE * 50).batch(SSL_BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(process_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(buffer_size=SSL_BATCH_SIZE * 10).batch(SSL_BATCH_SIZE).prefetch(
        buffer_size=tf.data.AUTOTUNE)

    #val_ds = tf.data.Dataset.from_tensor_slices((val_files))
    #val_ds = val_ds.map(process_image_only, num_parallel_calls=tf.data.AUTOTUNE)
    #val_ds = val_ds.batch(ssl_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    #Get feature extractor
    #encoder = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE)
    encoder = get_backbone(USE_BACKBONE, input_shape=IMAGE_SIZE, weights='imagenet')
    #Encoder.trainable = True
    #Encoder.summary()

    #pretrained_model_file = f'run_view_with_noise_Aug-True_T03_supcon/ap20_pl20_ps20_do20_mm20/ssl/best_model.h5'
    #pretrained_model = tf.keras.models.load_model(pretrained_model_file)
    #encoder = pretrained_model.layers[1]

    #AUG_MIN_AREA = 0.8
    #AUG_ROTATION = 0.10
    #aug = get_augmenter_echo(IMAGE_SIZE, AUG_MIN_AREA, AUG_ROTATION)
    model = Contrastive_Model(encoder=encoder)
    model.compile(contrastive_optimizer=tf.keras.optimizers.Adam(1e-4))  #1e-4 for convnext and swintransformer otherwise default param

    #COSINE DECAY
    #EPOCHS = 50
    #STEPS_PER_EPOCH = num_train // SSL_BATCH_SIZE
    #TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    #WARMUP_EPOCHS = int(EPOCHS * 0.10)
    #WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)
    #lr_decayed_fn = WarmUpCosine(
    #learning_rate_base=1e-3,
    #total_steps=EPOCHS * STEPS_PER_EPOCH,
    #warmup_learning_rate=0.0,
    #warmup_steps=WARMUP_STEPS
    #)

    callbks = []
    #Early stopping
    #early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #patience=10, start_from_epoch=20,
    #restore_best_weights=True)
    #Callbks.append(early_stopper)

    #Model checkpoint to save best weights
    #model_save_path = os.path.join(output_folder, 'model_ep_{epoch}.h5')
    #model_chpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
    #monitor='loss',
    #verbose=1,
    #save_weights_only=False,
    #save_best_only=False,
    #save_freq='epoch',
    #period=10
    #)
    #Callbks.append(model_chpt)
    #Callbks.append(SelectiveModelCheckpoint(output_folder=output_folder))

    #Model checkpoint to save best weights
    model_save_path_encoder = os.path.join(output_folder, 'best_model.h5')
    model_chpt1 = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path_encoder,
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_weights_only=False,
                                                     save_best_only=True,
                                                     save_freq='epoch',
                                                     #period=5
                                                     )
    callbks.append(model_chpt1)

    #Csv logger
    csv_save_path = os.path.join(output_folder, f'epoch_history.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_save_path)
    callbks.append(csv_logger)

    print(f'OBJECTIVE: {USE_LOSS}')
    print(f'TEMPERATURE: {TEMPERATURE}')
    print(f'DROP_K: {DROP_K}')

    start = time.time()

    history = model.fit(
        train_ds,
        epochs=24,
        validation_data=val_ds,
        callbacks=callbks
    )

    end = time.time()
    ellapsed_time = end - start

    #model_weights_save_path_last_epoch = os.path.join(output_folder, 'ssl_encoder_model_weights.h5')
    #Model.encoder.save_weights(model_weights_save_path_last_epoch)
    #model_save_path_last_epoch = os.path.join(output_folder, 'ssl_encoder_model.h5')
    #Model.encoder.save(model_save_path_last_epoch)
    #model_weights_save_path_last_epoch = os.path.join(output_folder, 'ssl_supercon_model_weights.h5')
    #Model.supercon_model.save_weights(model_weights_save_path_last_epoch)
    #model_save_path_last_epoch = os.path.join(output_folder, 'ssl_supercon_model.h5')
    #Model.supercon_model.save(model_save_path_last_epoch)

    tf.keras.backend.clear_session()

    return history, ellapsed_time

def get_backbone(backbone, input_shape=(224, 224, 3), weights = "imagenet"):
    b = str(backbone).lower()

    if b == "xception":
        return tf.keras.applications.Xception(include_top=False, weights=weights, input_shape=input_shape)

    elif b == "resnet50":
        return tf.keras.applications.ResNet50(include_top=False, weights=weights, input_shape=input_shape)

    elif b == "resnet101":
        return tf.keras.applications.ResNet101(include_top=False, weights=weights, input_shape=input_shape)

    elif b == "densenet121":
        return tf.keras.applications.DenseNet121(include_top=False, weights=weights, input_shape=input_shape)

    elif b == "convnextbase":
        return tf.keras.applications.ConvNeXtBase(include_top=False, weights=weights, input_shape=input_shape)
        #Use batch_size = 32
        #Use AdamW lr=1e-4
    elif b == "convnexttiny":
        return tf.keras.applications.ConvNeXtTiny(include_top=False, weights=weights, input_shape=input_shape)

    elif b == "efficientnetv2s":
        return tf.keras.applications.EfficientNetV2S(include_top=False, weights=weights, input_shape=input_shape)

    elif b == 'vit_base':
        import timm
        from keras_cv_attention_models import beit
        pretrained_weights = (weights == "imagenet")
        torch_model = timm.create_model("vit_base_patch16_224", pretrained=pretrained_weights).eval()
        #For vit_tiny_patch16_224: embed_dim=192, depth=12, num_heads=3, patch_size=16
        #For vit_base_patch16_224: embed_dim=768, depth=12, num_heads=12, patch_size=16
        encoder = beit.ViT(
            input_shape=(224, 224, 3),
            num_classes=0,  #Backbone only (no classifier head)
            patch_size=16,
            embed_dim=768, depth=12, num_heads=12,  #<-- change to 768/12/12 for "base"
        )
        beit.keras_model_load_weights_from_pytorch_model(encoder, torch_model)
        return encoder
        #Use AdamW lr=1e-4, batch_size 32

    elif b == 'vit_small':
        import timm
        from keras_cv_attention_models import beit
        pretrained_weights = (weights == "imagenet")
        torch_model = timm.create_model("vit_small_patch16_224", pretrained=pretrained_weights).eval()
        #For vit_tiny_patch16_224: embed_dim=192, depth=12, num_heads=3, patch_size=16
        #For vit_base_patch16_224: embed_dim=768, depth=12, num_heads=12, patch_size=16
        encoder = beit.ViT(
            input_shape=(224, 224, 3),
            num_classes=0,  #Backbone only (no classifier head)
            patch_size=16,
            embed_dim=384, depth=12, num_heads=12,  #<-- change to 768/12/12 for "base"
        )
        beit.keras_model_load_weights_from_pytorch_model(encoder, torch_model)
        return encoder

    elif b=="swintransformerv2base":
        from keras_cv_attention_models import swin_transformer_v2
        encoder = swin_transformer_v2.SwinTransformerV2Base_window16(input_shape=IMAGE_SIZE, pretrained=weights,
                                                                     num_classes=0)
        return encoder
        #Use use AdamW lr=1e-4, batch_size 32
    elif b=="swintransformerv2tiny":
        from keras_cv_attention_models import swin_transformer_v2
        encoder = swin_transformer_v2.SwinTransformerV2Tiny_window16(input_shape=IMAGE_SIZE, pretrained=weights,
                                                                     num_classes=0)
        return encoder
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

def train_model(train_ds, val_ds, output_folder, percentage, run_index, use_image_net_weights):

    weights = None
    if use_image_net_weights:
        weights = 'imagenet'
        print('Using image net weights.')

    encoder = get_backbone(USE_BACKBONE, input_shape=IMAGE_SIZE, weights=weights)

    if USE_BACKBONE not in ['vit', 'vit_base', 'vit_tiny', 'vit_small']:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=IMAGE_SIZE))
        model.add(encoder)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=IMAGE_SIZE))
        model.add(tf.keras.layers.Rescaling(1. /255.0))
        model.add(encoder)
        model.add(tf.keras.layers.LayerNormalization(epsilon=1e-6)) #try removing this
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))


    opt = tf.keras.optimizers.Adam()
    if 'vit' in USE_BACKBONE or 'swintransformer' in USE_BACKBONE or 'convnext' in USE_BACKBONE:
        opt = tf.keras.optimizers.AdamW(learning_rate=1e-4)

    #COSINE DECAY
    #EPOCHS = 100
    #STEPS_PER_EPOCH = num_train // BATCH_SIZE
    #TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    #WARMUP_EPOCHS = int(EPOCHS * 0.10)
    #WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)
    #lr_decayed_fn = WarmUpCosine(
    #learning_rate_base=1e-3,
    #total_steps=EPOCHS * STEPS_PER_EPOCH,
    #warmup_learning_rate=0.0,
    #warmup_steps=WARMUP_STEPS
    #)
    #EXPONENTIAL DECAY
    #epochs = 100
    #initial_learning_rate = 1e-3
    #final_learning_rate = 1e-5
    #learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    #steps_per_epoch = int(num_train / BATCH_SIZE)
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate=initial_learning_rate,
    #decay_steps=steps_per_epoch,
    #decay_rate=learning_rate_decay_factor,
    #staircase=True)
    #

    model.build()

    model.summary()

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'], #run_eagerly=True
    )

    callbks = []
    #Early stopping
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=5, start_from_epoch=20,
                                                     restore_best_weights=True)
    callbks.append(early_stopper)
    #Model checkpoint to save best weights
    model_save_path = os.path.join(output_folder, f'model_run{run_index}_perc{percentage}.h5')
    model_chpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    )
    callbks.append(model_chpt)
    #Csv logger
    csv_save_path = os.path.join(output_folder, f'epoch_history_run{run_index}_perc{percentage}.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_save_path)
    callbks.append(csv_logger)

    history = model.fit(
        train_ds,
        epochs=200,
        validation_data=val_ds,
        callbacks = callbks,
    )

    tf.keras.backend.clear_session()

    return model, history


def fine_tune_model(train_ds, val_ds, output_folder, percentage, run_index, pretrained_model_file, num_train, linear_eval):
    #pretrained_model_file = f'run_view_with_noise_Aug-True_T03_supcon/ap20_pl20_ps20_do20_mm20/ssl/best_model.h5' #TODO

    custom_objects = {
        name: obj
        for name, obj in A.__dict__.items()
        if inspect.isclass(obj) and issubclass(obj, tf.keras.layers.Layer)
    }
    custom_objects["LayerScale"] = LayerScale

    pretrained_model = tf.keras.models.load_model(pretrained_model_file, custom_objects=custom_objects)

    encoder =pretrained_model
    if linear_eval:
        encoder.trainable = False

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=IMAGE_SIZE))
    model.add(encoder)
    #Model.add(tf.keras.layers.Flatten())
    #Model.add(tf.keras.layers.Dense(256, activation='relu'))
    #Model.add(tf.keras.layers.BatchNormalization())
    #Model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    #COSINE DECAY
    #EPOCHS = 20
    #STEPS_PER_EPOCH = num_train // BATCH_SIZE
    #TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    #WARMUP_EPOCHS = 4#int(EPOCHS * 0.10)
    #WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)
    #lr_decayed_fn = WarmUpCosine(
    #learning_rate_base=1e-4,
    #total_steps=EPOCHS * STEPS_PER_EPOCH,
    #warmup_learning_rate=0.0,
    #warmup_steps=WARMUP_STEPS
    #)
    #EXPONENTIAL DECAY
    #epochs = 100
    #initial_learning_rate = 1e-3
    #final_learning_rate = 1e-5
    #learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    #steps_per_epoch = int(num_train / BATCH_SIZE)
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate=initial_learning_rate,
    #decay_steps=steps_per_epoch,
    #decay_rate=learning_rate_decay_factor,
    #staircase=True)
    #

    model.build()

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'], #run_eagerly=True
    )

    callbks = []
    #Early stopping
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=5, start_from_epoch=10,
                                                     restore_best_weights=True)
    callbks.append(early_stopper)
    #Model checkpoint to save best weights
    model_save_path = os.path.join(output_folder, f'model_run{run_index}_perc{percentage}.h5')
    model_chpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    )
    callbks.append(model_chpt)
    #Csv logger
    csv_save_path = os.path.join(output_folder, f'epoch_history_run{run_index}_perc{percentage}.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_save_path)
    callbks.append(csv_logger)

    if linear_eval:
        history = model.fit(
            train_ds,
            epochs=1,
            validation_data=val_ds,
            callbacks=callbks,
        )
    else:
        history = model.fit(
            train_ds,
            epochs=200,
            validation_data=val_ds,
            callbacks = callbks,
        )

    tf.keras.backend.clear_session()

    return model, history


def save_confusion_matrix(test_files, y_true, y_pred, run, class_lookup, output_folder, percentage, threshold=4):
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_2(cm, list(class_lookup.values()), normalize=False, title='Confusion Matrix')
    plt.savefig(os.path.join(output_folder, f'confusion_matrix_run{run}_perc{percentage}.png'))

    return #TODO
    sub_folder = os.path.join(output_folder, f'confusion info_{run}_{percentage}')
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    row_list = [["True Label", "Pred Label", "Hard Cases"]]
    num_classes = len(class_lookup)
    for i in range(num_classes):
        for j in range(num_classes):
            if i==j: continue

            true_label = class_lookup[i]
            pred_label = class_lookup[j]

            val = cm[i, j]
            if val>=threshold:
                folder = os.path.join(sub_folder, f'{true_label} __ {pred_label} __ {val}')
                if not os.path.exists(folder):
                    os.makedirs(folder)
                print(f'{true_label} - {pred_label} - {val}')
                row_list.append([true_label, pred_label, val])

                hard_cases = []
                class_files = []
                other_files = []
                for k in range(len(test_files)):
                    file = test_files[k]
                    if y_true[k]==i:#if true_label in file:
                        if y_true[k]==i and y_pred[k] == j:
                            target = os.path.basename(file)
                            shutil.copyfile(file, os.path.join(folder, target))
                            hard_cases.append(file)
                        elif y_true[k]==i and y_pred[k] == i:
                            class_files.append(file)
                    elif pred_label in file:
                        other_files.append(file)

                #Choose some random images from true_label folder, pred_folder, and hard_cases.
                #Plot them in rows for comparison
                num_per_row = 4
                plt.figure(figsize=(20, 16), dpi=300)
                cnt = 1

                if num_per_row > threshold:
                    num_per_row=threshold

                rand_list = random.sample(range(len(class_files)), num_per_row)
                for i in range(num_per_row):
                    img = cv2.imread(class_files[rand_list[i]])
                    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_LINEAR)
                    ax = plt.subplot(3, 4, cnt)
                    if i==1:
                        ax.set_title(f'Examples of {true_label.upper()}', {'fontsize': 16}, 'right')
                    plt.imshow(img)
                    plt.axis('off')
                    cnt+=1

                rand_list = random.sample(range(len(hard_cases)), num_per_row)
                for i in range(num_per_row):
                    img = cv2.imread(hard_cases[rand_list[i]])
                    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_LINEAR)
                    ax = plt.subplot(3, 4, cnt)
                    if i==1:
                        ax.set_title(f'Examples of {true_label.upper()} incorrectly classified as {pred_label.upper()}', {'fontsize': 16}, 'right')
                    plt.imshow(img)
                    plt.axis('off')
                    cnt+=1

                rand_list = random.sample(range(len(other_files)), num_per_row)
                for i in range(num_per_row):
                    img = cv2.imread(other_files[rand_list[i]])
                    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_LINEAR)
                    ax = plt.subplot(3, 4, cnt)
                    if i==1:
                        ax.set_title(f'Examples of {pred_label.upper()}', {'fontsize': 16}, 'right')
                    plt.imshow(img)
                    plt.axis('off')
                    cnt += 1


                plt.tight_layout()
                plt.savefig(os.path.join(os.path.join(sub_folder, f'{true_label}__{pred_label}__{val}.png')), dpi=300, bbox_inches='tight')
                plt.clf()

    all_incorrect = []
    for i in range(len(test_files)):
        if y_true[i]!=y_pred[i]:
            basename = os.path.basename(test_files[i])
            all_incorrect.append(basename)

    with open(os.path.join(sub_folder, 'all_inorrect.txt'), 'w') as f:
        for filename in all_incorrect:
            f.write(f"{filename}\n")

    with open(os.path.join(sub_folder, 'info.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)

from keras.saving import register_keras_serializable
from keras.utils import custom_object_scope

@register_keras_serializable(package="keras.applications.convnext")
class LayerScale(tf.keras.layers.Layer):
    def __init__(
        self,
        init_values=1e-6,
        projection_dim=None,   #Keras ConvNeXt passes this in some versions
        axis=-1,
        **kwargs               #Tolerate extra keys from other versions
    ):
        super().__init__(**kwargs)
        self.init_values = float(init_values)
        self.projection_dim = None if projection_dim is None else int(projection_dim)
        self.axis = int(axis)
        #Store any extra kwargs to preserve config round-trips (optional)
        self._extra_config = {k: v for k, v in kwargs.items() if k not in ("name", "trainable", "dtype")}

    def build(self, input_shape):
        if self.projection_dim is not None:
            dim = self.projection_dim
        else:
            #Infer from input shape along `axis` (e.g., channels-last: -1)
            dim = int(input_shape[self.axis])
        self.gamma = self.add_weight(
            name="gamma",
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        #Broadcast multiply over spatial dims: (B,H,W,C) * (C,)
        return x * self.gamma

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "init_values": self.init_values,
            "projection_dim": self.projection_dim,
            "axis": self.axis,
            **self._extra_config
        })
        return cfg

def evaluate(save_folder, class_lookup, model_file, test_files, test_labels, run, percentage, name):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    test_images_ds = tf.data.Dataset.from_tensor_slices((test_files))
    test_images_ds = test_images_ds.map(process_image_only, num_parallel_calls=tf.data.AUTOTUNE)
    test_images_ds = test_images_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    from keras_cv_attention_models import attention_layers
    #custom_objs = {
    #"LayerScale": LayerScale,
    #"PatchConv2DWithResampleWeights": attention_layers.PatchConv2DWithResampleWeights,
    #"PatchConv2D": attention_layers.PatchConv2D, #sometimes also present
    #}
    #with custom_object_scope(custom_objs):
    #model = tf.keras.models.load_model(model_file, compile=False)

    custom_objects = {
        name: obj
        for name, obj in A.__dict__.items()
        if inspect.isclass(obj) and issubclass(obj, tf.keras.layers.Layer)
    }
    custom_objects["LayerScale"] = LayerScale

    model = tf.keras.models.load_model(model_file, compile=False, custom_objects = custom_objects)

    y_true = test_labels
    y_pred_prob = model.predict(test_images_ds)
    y_pred = y_pred_prob.argmax(axis=-1)

    acc = accuracy_score(y_true, y_pred)
    print(f" Accuracy {name}: {acc}")
    #This gives metrics per class
    precision, recall, f1score, support = score(y_true, y_pred)
    #Average all
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_avg = np.mean(f1score)

    save_confusion_matrix(test_files, y_true, y_pred, run, class_lookup, save_folder, percentage)

    cm = confusion_matrix(y_true, y_pred)
    #Acc per class:
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    acc_per_class_dict = {}
    for idx, accy in enumerate(acc_per_class):
        acc_per_class_dict[class_lookup[idx]] = acc_per_class[idx]
    utils.write_dict_to_json(acc_per_class_dict, save_folder, f'classification_report_acc_run{run}.json')

    #Report
    try:
        #Determine which labels are actually present
        present_labels = sorted(set(y_true) | set(y_pred))  #Union of both sets
        #Get class names for present labels only
        present_class_names = [class_lookup[i] for i in present_labels]
        class_report = classification_report(y_true, y_pred, labels=present_labels, target_names=present_class_names, output_dict=True)
        utils.write_dict_to_json(class_report, save_folder, f'classification_report_run{run}.json')
    except Exception as e:
        print(f"[ERROR] Failed to generate or save classification report: {e}")

    results_dict = {}

    results_dict['num_classes'] = len(class_lookup)
    results_dict['acc'] = acc
    results_dict['precision'] = precision_avg
    results_dict['recall'] = recall_avg
    results_dict['f1'] = f1_avg
    utils.write_dict_to_json(results_dict, save_folder, f'results_{run}.json')
    utils.write_dict_to_json({'files': test_files, 'y_true':y_true,'y_pred':y_pred.tolist(),'y_pred_prob':y_pred_prob.tolist()}, save_folder, f'all_predictions_{run}.json')

def evaluate_multi(save_folder, class_lookup, model_file, test_files, labels_exp1, labels_exp2, labels_exp3, run, percentage, name):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    test_images_ds = tf.data.Dataset.from_tensor_slices((test_files))
    test_images_ds = test_images_ds.map(process_image_only, num_parallel_calls=tf.data.AUTOTUNE)
    test_images_ds = test_images_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(model_file)
    y_pred_prob = model.predict(test_images_ds)
    y_pred = y_pred_prob.argmax(axis=-1)

    test_labels = [
        int(pred) if pred in (exp1, exp2, exp3) else int(exp1)
        for pred, exp1, exp2, exp3 in zip(y_pred, labels_exp1, labels_exp2, labels_exp3)
    ]
    y_true = test_labels

    acc = accuracy_score(y_true, y_pred)
    print(f" Accuracy {name}: {acc}")
    #This gives metrics per class
    precision, recall, f1score, support = score(y_true, y_pred)
    #Average all
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_avg = np.mean(f1score)

    save_confusion_matrix(test_files, y_true, y_pred, run, class_lookup, save_folder, percentage)

    cm = confusion_matrix(y_true, y_pred)
    #Acc per class:
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    acc_per_class_dict = {}
    for idx, accy in enumerate(acc_per_class):
        acc_per_class_dict[class_lookup[idx]] = acc_per_class[idx]
    utils.write_dict_to_json(acc_per_class_dict, save_folder, f'classification_report_acc_run{run}.json')

    #Report
    try:
        #Determine which labels are actually present
        present_labels = sorted(set(y_true) | set(y_pred))  #Union of both sets
        #Get class names for present labels only
        present_class_names = [class_lookup[i] for i in present_labels]
        class_report = classification_report(y_true, y_pred, labels=present_labels, target_names=present_class_names, output_dict=True)
        utils.write_dict_to_json(class_report, save_folder, f'classification_report_run{run}.json')
    except Exception as e:
        print(f"[ERROR] Failed to generate or save classification report: {e}")

    results_dict = {}

    results_dict['num_classes'] = len(class_lookup)
    results_dict['acc'] = acc
    results_dict['precision'] = precision_avg
    results_dict['recall'] = recall_avg
    results_dict['f1'] = f1_avg
    utils.write_dict_to_json(results_dict, save_folder, f'results_{run}.json')
    utils.write_dict_to_json({'files': test_files, 'y_true':y_true,'y_pred':y_pred.tolist(),'y_pred_prob':y_pred_prob.tolist()}, save_folder, f'all_predictions_{run}.json')


def save_cluster_analysis_results(mismatched_points, train_files, cluster_info, class_lookup, output_folder):
    mismatched_txt = os.path.join(output_folder, 'mismatched_files.txt')
    with open(mismatched_txt, 'w') as f:
        for file_path in mismatched_points:
            f.write(file_path + '\n')

    summary_txt = os.path.join(output_folder, 'cluster_summary.txt')
    with open(summary_txt, 'w') as f:
        for label, info in cluster_info.items():
            f.write(f"Cluster Label: {label} ({class_lookup[label]})\n")
            f.write(f"  Centroid: {info['centroid']}\n")
            f.write(f"  Radius: {info['radius']:.4f}\n")
            f.write(f"  Dominant Label: {info['dominant_label']} ({class_lookup[info['dominant_label']]})\n")
            f.write(f"  Label Distribution:\n")
            for lbl, count in info['label_distribution'].items():
                f.write(f"    {lbl} ({class_lookup[lbl]}): {count}\n")
            f.write(f"  Num Points in Radius: {info['num_points']}\n")
            f.write("\n")


def plot_tsne_with_clusters(x, y, labels_array, cluster_info, class_lookup, mismatched_points, train_files, output_folder):
    plt.figure(figsize=(12, 10))
    coords = np.vstack((x, y)).T
    label_names = [class_lookup[lbl] for lbl in labels_array]
    scatter = plt.scatter(x, y, c=labels_array, cmap='tab20', alpha=0.6, s=10)

    for label, info in cluster_info.items():
        cx, cy = info['centroid']
        plt.scatter(cx, cy, marker='X', c='black', s=100)
        plt.text(cx, cy, class_lookup[label], fontsize=8, weight='bold', ha='center', va='center')

    mismatched_idxs = [train_files.index(fp) for fp in mismatched_points if fp in train_files]
    if mismatched_idxs:
        plt.scatter(x[mismatched_idxs], y[mismatched_idxs], c='red', marker='x', s=30, label='Mismatch')

    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE Clusters with Centroids and Mismatches")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'tsne_clusters.png'), dpi=300)
    plt.close()

def plot_tsne_old(x, y, color_list, labels_list, save_file):

    plt.clf()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot()
    scatter = ax.scatter(x, y, c=color_list, s=1.5)
    #Ax.legend(loc='best')
    plt.legend(handles=scatter.legend_elements()[0], labels=labels_list)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')


def plot_tsne(x, y, color_list, labels_list, save_file):
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)  #Increased width for better spacing

    #Use tab20 colormap
    num_colors = len(set(color_list))
    cmap = cm.get_cmap('tab20', num_colors)

    #Define marker styles
    marker_styles = ['o', 's', '*', 'd', '^', 'P']

    #Plot points with the color list and different marker styles
    for i, label in enumerate(set(color_list)):
        indices = np.where(np.array(color_list) == label)
        marker = marker_styles[i % len(marker_styles)]
        ax.scatter(x[indices], y[indices], c=[cmap(i)], s=6, marker=marker, alpha=0.8, edgecolors='none')

    #Adding axis labels for clarity
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE Clusters Visualization")

    #Create a legend with class labels, positioned to the right
    legend_handles = [
        plt.Line2D([0], [0], marker=marker_styles[i % len(marker_styles)], color='w', markerfacecolor=cmap(i),
                   markersize=10, linestyle='None') for i in range(num_colors)]
    ax.legend(legend_handles, labels_list, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium', ncol=2)

    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne_with_separate_legends(
        x, y,                  #Embeddings (np.ndarray)
        color_list,            #List/array of same length as x/y giving a color-index per point
        labels_list,           #List of human-readable names, same order as class indices
        plot_file,             #Main scatter filename
        legend_vert_file,      #Legend file (vertical list, 1 column)
        legend_horiz_file,     #Legend file (horizontal list, many columns)
        marker_styles=('o', 's', '*', 'd', '^', 'P'),  #Markers to cycle through
        dpi=300
    ):
    """
    Saves three PNGs: the scatter plot without legend, a vertical legend, and a horizontal legend.

    Uses the same `tab20`-based colormap as:
        cmap = cm.get_cmap('tab20', num_colors)

    but reorders which class gets which color so that, within each group
    (apical / plax / psax / sub / mmode / doppler), you do not get the
    "light/dark" variants of the same hue.
    """
    #------------------------------------------------------------------
    #0) Build mapping: class index -> group, and compute round-robin order
    #------------------------------------------------------------------
    num_classes = len(labels_list)

    #Helper: which high-level group a label belongs to
    def get_group(label_name):
        for g, mapping in groupings_dict.items():
            if label_name in mapping:
                return g
        #Special alias used earlier
        if label_name == "doppler-ao":
            return "doppler"
        return "other"

    #Collect class indices per group
    group_order = ["apical", "plax", "psax", "sub", "mmode", "doppler", "other"]
    group_to_ids = {g: [] for g in group_order}

    for class_idx, name in enumerate(labels_list):
        g = get_group(name)
        if g not in group_to_ids:
            group_to_ids[g] = []
            group_order.append(g)
        group_to_ids[g].append(class_idx)

    #Round-robin through groups to get a *color order* of class indices.
    #This makes sure indices assigned to a given group are spaced apart
    #In the colormap (so you don't get adjacent "light/dark" variants).
    ordered_class_ids = []
    still_left = True
    while still_left:
        still_left = False
        for g in group_order:
            if group_to_ids[g]:
                ordered_class_ids.append(group_to_ids[g].pop(0))
                still_left = True

    #One discrete color per class, using the same call you already had
    num_colors = len(ordered_class_ids)
    cmap = cm.get_cmap('tab20', num_colors)

    #Map each class index -> color index in the colormap
    class_to_color_index = {cls: i for i, cls in enumerate(ordered_class_ids)}

    #------------------------------------------------------------------
    #1) Build main scatter without legend
    #------------------------------------------------------------------
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)

    color_array = np.array(color_list)

    #Scatter points class by class
    for class_idx in range(num_classes):
        idx = np.where(color_array == class_idx)[0]
        if idx.size == 0:
            continue  #class not present in this embedding

        color_idx = class_to_color_index[class_idx]
        marker = marker_styles[class_idx % len(marker_styles)]

        ax.scatter(
            x[idx], y[idx],
            c=[cmap(color_idx)],
            s=10,
            marker=marker,
            alpha=0.8,
            edgecolors='none'
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=20)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=20)
    #Remove ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    #------------------------------------------------------------------
    #2) Prepare legend handles (same mapping as scatter)
    #------------------------------------------------------------------
    handles = []
    for class_idx, name in enumerate(labels_list):
        color_idx = class_to_color_index[class_idx]
        marker = marker_styles[class_idx % len(marker_styles)]
        handle = plt.Line2D(
            [0], [0],
            marker=marker,
            color='w',
            markerfacecolor=cmap(color_idx),
            markersize=14,
            linestyle='None'
        )
        handles.append(handle)

    num_colors = len(handles)

    #------------------------------------------------------------------
    #3) Vertical legend (single column)
    #------------------------------------------------------------------
    fig_legend_v = plt.figure(figsize=(3, 0.5 + 0.35 * num_colors), dpi=dpi)
    fig_legend_v.legend(
        handles, labels_list,
        loc='center',
        ncol=1,
        frameon=False,
        fontsize='medium'
    )
    fig_legend_v.tight_layout()
    fig_legend_v.savefig(legend_vert_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig_legend_v)

    #------------------------------------------------------------------
    #4) Horizontal legend (many columns)
    #------------------------------------------------------------------
    ncol = min(5, num_colors)
    nrows = int(np.ceil(num_colors / ncol))
    fig_legend_h = plt.figure(figsize=(0.8 * num_colors, 0.8 * nrows), dpi=dpi)
    fig_legend_h.legend(
        handles, labels_list,
        loc='center',
        ncol=ncol,
        frameon=False,
        fontsize='medium'
    )
    fig_legend_h.tight_layout()
    fig_legend_h.savefig(legend_horiz_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig_legend_h)


def plot_tsne_with_separate_legends1(
        x, y,                  #Embeddings (np.ndarray)
        color_list,            #List/array of same length as x/y giving a color-index per point
        labels_list,           #List of human-readable names, same order as unique values in color_list
        plot_file,             #Main scatter filename
        legend_vert_file,      #Legend file (vertical list, 1 column)
        legend_horiz_file,     #Legend file (horizontal list, many columns)
        marker_styles=('o', 's', '*', 'd', '^', 'P'),  #Markers to cycle through
        dpi=300
    ):
    """
    Saves three PNGs: the scatter plot without legend, a vertical legend, and a horizontal legend.
    """
    #------------------------------------------------------------------
    #1) Build main scatter without legend
    #------------------------------------------------------------------
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)

    #Map each unique label to an index
    unique_labels = list(dict.fromkeys(color_list))  #Preserves order
    num_colors = len(unique_labels)
    cmap = cm.get_cmap('tab20', num_colors)

    #Scatter each label group
    for i, lbl in enumerate(unique_labels):
        idx = np.where(np.array(color_list) == lbl)
        marker = marker_styles[i % len(marker_styles)]
        ax.scatter(x[idx], y[idx],
                   c=[cmap(i)], s=10, marker=marker,
                   alpha=0.8, edgecolors='none')

    ax.set_xlabel("t-SNE Dimension 1", fontsize=20)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=20)
    #Ax.set_title("t-SNE Clusters Visualization")

    #Hide x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    #Optionally remove tick labels as well (redundant if no ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #Tight layout without legend
    plt.tight_layout()
    plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    #------------------------------------------------------------------
    #2) Prepare legend handles (re-use same mapping)
    #------------------------------------------------------------------
    handles = [
        plt.Line2D([0], [0],
            marker=marker_styles[i % len(marker_styles)],
            color='w', markerfacecolor=cmap(i),
            markersize=14, linestyle='None')
        for i in range(num_colors)
    ]

    #------------------------------------------------------------------
    #3) Stand-alone *vertical* legend (1 column)
    #------------------------------------------------------------------
    fig_legend_v = plt.figure(figsize=(3, 0.5 + 0.35 * num_colors), dpi=dpi)
    fig_legend_v.legend(
        handles, labels_list,
        loc='center', ncol=1, frameon=False,
        fontsize='medium'
    )
    fig_legend_v.tight_layout()
    fig_legend_v.savefig(legend_vert_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig_legend_v)

    #------------------------------------------------------------------
    #4) Stand-alone *horizontal* legend (many columns, 1-2 rows)
    #We pick ncol = min(5, num_colors) so long legends wrap nicely.
    #------------------------------------------------------------------
    ncol = min(5, num_colors)  #Tweak as desired
    nrows = int(np.ceil(num_colors / ncol))
    fig_legend_h = plt.figure(figsize=(0.8 * num_colors, 0.8 * nrows), dpi=dpi)
    fig_legend_h.legend(
        handles, labels_list,
        loc='center', ncol=ncol, frameon=False,
        fontsize='medium'
    )
    fig_legend_h.tight_layout()
    fig_legend_h.savefig(legend_horiz_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig_legend_h)


def analyze_tsne_clusters_with_dbscan(x, y, labels_array, file_paths, class_lookup, output_folder, eps=5.0, min_samples=5):
    coords = np.vstack((x, y)).T
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    cluster_labels = db.labels_

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of clusters found by DBSCAN: {n_clusters}")

    cluster_info = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id != -1:
            cluster_info[cluster_id].append(i)

    centroids = {}
    for cluster_id, indices in cluster_info.items():
        cluster_coords = coords[indices]
        centroid = np.mean(cluster_coords, axis=0)
        centroids[cluster_id] = centroid

    mismatches = []
    for cluster_id, indices in cluster_info.items():
        true_labels = labels_array[indices]
        label_counts = np.bincount(true_labels, minlength=len(class_lookup))
        dominant_label = np.argmax(label_counts)

        for i in indices:
            true_label = labels_array[i]
            if true_label != dominant_label:
                mismatches.append({
                    "file": file_paths[i],
                    "true_label": class_lookup[true_label],
                    "cluster_dominant_label": class_lookup[dominant_label],
                    "cluster_id": int(cluster_id)
                })

    #Save mismatches
    mismatch_df = pd.DataFrame(mismatches)
    mismatch_file = os.path.join(output_folder, "tsne_dbscan_mismatches.csv")
    mismatch_df.to_csv(mismatch_file, index=False)

    #Save summary JSON
    summary = {
        "n_clusters": n_clusters,
        "n_mismatches": len(mismatches)
    }
    with open(os.path.join(output_folder, "tsne_dbscan_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    #Plot
    plt.figure(figsize=(12, 10))
    unique_clusters = sorted(set(cluster_labels))
    colors = plt.cm.get_cmap("tab20", len(unique_clusters))

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        if cluster_id == -1:
            plt.scatter(x[cluster_mask], y[cluster_mask], c="gray", label="Noise", alpha=0.4, s=10)
        else:
            plt.scatter(x[cluster_mask], y[cluster_mask], label=f"Cluster {cluster_id}", alpha=0.6, s=10)

    #Plot centroids
    for cluster_id, centroid in centroids.items():
        plt.plot(centroid[0], centroid[1], 'kx', markersize=12, markeredgewidth=2)

    plt.legend(markerscale=2)
    plt.title("DBSCAN Clusters on t-SNE Projection")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "tsne_dbscan_clusters.png"), dpi=300)
    plt.close()

def analyze_tsne_clusters_with_kmeans(x, y, labels_array, file_paths, class_lookup, output_folder, n_clusters=47):
    coords = np.vstack((x, y)).T

    #Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(coords)

    #Find centroids
    centroids = kmeans.cluster_centers_

    mismatches = []
    cluster_info = defaultdict(list)

    for i, cluster_id in enumerate(cluster_labels):
        cluster_info[cluster_id].append(i)

    for cluster_id, indices in cluster_info.items():
        true_labels = labels_array[indices]
        label_counts = np.bincount(true_labels, minlength=len(class_lookup))
        dominant_label = np.argmax(label_counts)

        for i in indices:
            true_label = labels_array[i]
            if true_label != dominant_label:
                mismatches.append({
                    "file": file_paths[i],
                    "true_label": class_lookup[true_label],
                    "cluster_dominant_label": class_lookup[dominant_label],
                    "cluster_id": int(cluster_id)
                })

    #Save mismatches
    mismatch_df = pd.DataFrame(mismatches)
    mismatch_file = os.path.join(output_folder, "tsne_kmeans_mismatches.csv")
    mismatch_df.to_csv(mismatch_file, index=False)

    #Save summary JSON
    summary = {
        "n_clusters": n_clusters,
        "n_mismatches": len(mismatches)
    }
    with open(os.path.join(output_folder, "tsne_kmeans_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    #Plot
    plt.figure(figsize=(12, 10))
    cmap = plt.cm.get_cmap("tab20", n_clusters)

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        plt.scatter(x[cluster_mask], y[cluster_mask],
                    color=cmap(cluster_id),
                    label=f"Cluster {cluster_id}", alpha=0.6, s=10)

    #Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', color='k', s=100, linewidths=3, label='Centroids')

    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("KMeans Clusters (47) on t-SNE Projection")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "tsne_kmeans_clusters.png"), dpi=300, bbox_inches='tight')
    plt.close()

def extract_features_and_run_tsne(pretraining_file, model_file, output_folder, class_lookup):
    label_names_list = list(class_lookup.values())
    pretraining_dict = utils.load_json(pretraining_file)
    train_files = pretraining_dict['train']
    train_labels = pretraining_dict['labels']

    train_files = train_files[:10000]
    train_labels = train_labels[:10000]

    #Add function call for tsne, and select new labels

    encoder = tf.keras.models.load_model(model_file)
    encoder.summary()

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.map(process_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=SSL_BATCH_SIZE * 10).batch(SSL_BATCH_SIZE).prefetch(
        buffer_size=tf.data.AUTOTUNE)

    #Extract features
    features_list = []
    labels_list = []

    for images, labels in train_ds:
        features = encoder(images, training=False)  #Shape: (batch_size, 7, 7, 2048)
        features = tf.reshape(features, [features.shape[0], -1])  #Shape: (batch_size, 100352)
        features_list.append(features.numpy())
        labels_list.append(labels.numpy())

    #Concatenate all batches
    features_array = np.concatenate(features_list, axis=0)  #Final shape: (total_samples, 100352)
    labels_array = np.concatenate(labels_list, axis=0)

    #Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    trans_dim = tsne.fit_transform(features_array)

    x = trans_dim[:, 0]
    y = trans_dim[:, 1]

    label_names_list = list(class_lookup.values())
    plot_tsne(x, y, labels_array, label_names_list, os.path.join(output_folder, 'tsne.png'))

    analyze_tsne_clusters_with_dbscan(x, y, labels_array, train_files, class_lookup, output_folder)
    analyze_tsne_clusters_with_kmeans(x, y, labels_array, train_files, class_lookup, output_folder)

#EMBEDDINGS ONLY#def analyze_tsne_with_kmeans(embeddings, labels_array, file_paths, class_lookup, output_folder, n_clusters=47, tsne_dim=2, perplexity=30, random_state=42):
    #Step 1: Run t-SNE
    print("Running t-SNE on embeddings...")
    tsne = TSNE(n_components=tsne_dim, perplexity=perplexity, random_state=random_state)
    reduced_embeddings = tsne.fit_transform(np.asarray(embeddings))

    #Step 2: KMeans on t-SNE output
    print("Running KMeans on t-SNE reduced data...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=random_state)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)

    centroids = kmeans.cluster_centers_
    cluster_info = defaultdict(list)
    for i, cid in enumerate(cluster_labels):
        cluster_info[cid].append(i)

    labels_array = np.array(labels_array)
    mismatches = []
    mismatches_dict = {}
    for cid, indices in cluster_info.items():
        true_labels = labels_array[indices]
        dominant_label = np.argmax(np.bincount(true_labels, minlength=len(class_lookup)))

        for i in indices:
            if labels_array[i] != dominant_label:
                val = {
                    "file": file_paths[i],
                    "gt_class_index": int(labels_array[i]),
                    "gt_label": class_lookup[labels_array[i]],
                    "cluster_dominant_class_index": int(dominant_label),
                    "cluster_dominant_label": class_lookup[dominant_label],
                    "cluster_id": int(cid)
                }
                mismatches.append(val)
                mismatches_dict[file_paths[i]] = val

    #Save mismatches and summary
    utils.write_dict_to_json(mismatches_dict, output_folder, "tsne_kmeans_mismatches.json")
    pd.DataFrame(mismatches).to_csv(os.path.join(output_folder, "tsne_kmeans_mismatches.csv"), index=False)
    json.dump({"n_clusters": n_clusters, "n_mismatches": len(mismatches)},
              open(os.path.join(output_folder, "tsne_kmeans_summary.json"), "w"), indent=2)

def analyze_tsne_with_hdbscan(embeddings, labels_array, file_paths, class_lookup, output_folder,
                               tsne_dim=2, perplexity=30, min_cluster_size=5, min_samples=5, random_state=42):
    print("Running t-SNE on embeddings...")
    tsne = TSNE(n_components=tsne_dim, perplexity=perplexity, random_state=random_state)
    reduced_embeddings = tsne.fit_transform(np.asarray(embeddings))

    print("Running HDBSCAN on t-SNE reduced data...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    labels_array = np.array(labels_array)
    cluster_info = defaultdict(list)
    for i, cid in enumerate(cluster_labels):
        if cid != -1:  #Ignore noise
            cluster_info[cid].append(i)

    mismatches = []
    mismatches_dict = {}
    for cid, indices in cluster_info.items():
        true_labels = labels_array[indices]
        dominant_label = np.argmax(np.bincount(true_labels, minlength=len(class_lookup)))

        for i in indices:
            if labels_array[i] != dominant_label:
                val = {
                    "file": file_paths[i],
                    "gt_class_index": int(labels_array[i]),
                    "gt_label": class_lookup[labels_array[i]],
                    "cluster_dominant_class_index": int(dominant_label),
                    "cluster_dominant_label": class_lookup[dominant_label],
                    "cluster_id": int(cid)
                }
                mismatches.append(val)
                mismatches_dict[file_paths[i]] = val

    #Save outputs
    pd.DataFrame(mismatches).to_csv(os.path.join(output_folder, "tsne_hdbscan_mismatches.csv"), index=False)
    with open(os.path.join(output_folder, "tsne_hdbscan_mismatches.json"), "w") as f:
        json.dump(mismatches_dict, f, indent=2)
    with open(os.path.join(output_folder, "tsne_hdbscan_summary.json"), "w") as f:
        json.dump({"n_clusters": len(cluster_info), "n_mismatches": len(mismatches)}, f, indent=2)

def analyze_embeddings_with_dbscan(embeddings, labels_array, file_paths, class_lookup, output_folder, eps=5.0, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    cluster_labels = db.labels_

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of clusters found by DBSCAN: {n_clusters}")

    cluster_info = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id != -1:
            cluster_info[cluster_id].append(i)

    labels_array = np.array(labels_array)
    #embeddings = np.array(embeddings)

    #centroids = {cid: np.mean(embeddings[indices], axis=0) for cid, indices in cluster_info.items()}

    mismatches = []
    for cluster_id, indices in cluster_info.items():
        true_labels = labels_array[indices]
        dominant_label = np.argmax(np.bincount(true_labels, minlength=len(class_lookup)))

        for i in indices:
            if labels_array[i] != dominant_label:
                mismatches.append({
                    "file": file_paths[i],
                    "true_label": class_lookup[labels_array[i]],
                    "cluster_dominant_label": class_lookup[dominant_label],
                    "cluster_id": int(cluster_id)
                })

    pd.DataFrame(mismatches).to_csv(os.path.join(output_folder, "embedding_dbscan_mismatches.csv"), index=False)
    json.dump({"n_clusters": n_clusters, "n_mismatches": len(mismatches)},
              open(os.path.join(output_folder, "embedding_dbscan_summary.json"), "w"), indent=2)


def analyze_embeddings_with_hdbscan(embeddings, labels_array, file_paths, class_lookup, output_folder, min_cluster_size=10, min_samples=5):
    #HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of clusters found by HDBSCAN: {n_clusters}")

    cluster_info = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id != -1:  #Ignore noise
            cluster_info[cluster_id].append(i)

    labels_array = np.array(labels_array)

    mismatches = []
    mismatches_dict = {}
    for cluster_id, indices in cluster_info.items():
        true_labels = labels_array[indices]
        dominant_label = np.argmax(np.bincount(true_labels, minlength=len(class_lookup)))

        for i in indices:
            if labels_array[i] != dominant_label:
                val = {
                    "file": file_paths[i],
                    "gt_class_index": int(labels_array[i]),  #If artificial noise has been injected, this wont be the GT label. It will be the noisy label.
                    "gt_label": class_lookup[labels_array[i]],  #If artificial noise has been injected, this wont be the GT label. It will be the noisy label.
                    "cluster_dominant_class_index": int(dominant_label),
                    "cluster_dominant_label": class_lookup[dominant_label],
                    "cluster_id": int(cluster_id)
                }
                mismatches.append(val)
                mismatches_dict[file_paths[i]] = val

    print(f"Number of mismatches found with HDBSCAN: {len(mismatches)}")

    #Save mismatches
    utils.write_dict_to_json(mismatches_dict, output_folder, "hdbscan_mismatches.json")
    pd.DataFrame(mismatches).to_csv(os.path.join(output_folder, "hdbscan_mismatches.csv"), index=False)
    json.dump({"n_clusters": n_clusters, "n_mismatches": len(mismatches)},
              open(os.path.join(output_folder, "hdbscan_summary.json"), "w"), indent=2)

def analyze_embeddings_with_kmeans(embeddings, labels_array, file_paths, class_lookup, output_folder, n_clusters=47):
    #kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42) #, init_size=5000
    cluster_labels = kmeans.fit_predict(embeddings)

    centroids = kmeans.cluster_centers_
    cluster_info = defaultdict(list)
    for i, cid in enumerate(cluster_labels):
        cluster_info[cid].append(i)

    dominance_ratio_threshold = 0.7
    distance_percentile = 98

    labels_array = np.array(labels_array)
    mismatches = []
    mismatches_dict = {}
    for cid, indices in cluster_info.items():
        true_labels = labels_array[indices]
        label_counts = np.bincount(true_labels, minlength=len(class_lookup))
        dominant_label = np.argmax(label_counts)
        dominant_count = label_counts[dominant_label]
        dominance_ratio = dominant_count / len(true_labels)

        #if dominance_ratio < dominance_ratio_threshold:
        #Continue #skip this cluster

        cluster_center = centroids[cid]
        distances = [np.linalg.norm(embeddings[i] - cluster_center) for i in indices]
        distance_threshold = np.percentile(distances, distance_percentile)

        for i in indices:
            if labels_array[i] != dominant_label:
                distance = np.linalg.norm(embeddings[i] - cluster_center)
                #if distance > distance_threshold:
                #Continue #too far, unreliable assignment

                val = {
                    "file": file_paths[i],
                    "gt_class_index": int(labels_array[i]),  #If artificial noise has been injected, this wont be the GT label. It will be the noisy label.
                    "gt_label": class_lookup[labels_array[i]],  #If artificial noise has been injected, this wont be the GT label. It will be the noisy label.
                    "cluster_dominant_class_index": int(dominant_label),
                    "cluster_dominant_label": class_lookup[dominant_label],
                    "cluster_id": int(cid)
                }
                mismatches.append(val)
                mismatches_dict[file_paths[i]] = val

    print(f"Number of mismatches found with Kmeans: {len(mismatches)}")

    utils.write_dict_to_json(mismatches_dict, output_folder, "kmeans_mismatches.json")
    pd.DataFrame(mismatches).to_csv(os.path.join(output_folder, "kmeans_mismatches.csv"), index=False)
    json.dump({"n_clusters": n_clusters, "n_mismatches": len(mismatches)},
              open(os.path.join(output_folder, "kmeans_summary.json"), "w"), indent=2)

    #Save per-class representative files (most typical by embedding)
    #find_best_example_per_class(
    #embeddings=embeddings,
    #labels_array=labels_array,
    #file_paths=file_paths,
    #class_lookup=class_lookup,
    #output_dir=os.path.join(output_folder, "class_representatives")
    #)

def extract_features_and_cluster_directly(pretraining_file, model_file, output_folder, class_lookup, use_kmeans=True, use_hdbscan=False, use_tsne_kmeans=True, use_tsne_hdbscan=True, n_clusters=47, eps=1.0, min_samples=10):
    pretraining_dict = utils.load_json(pretraining_file)
    train_files = pretraining_dict['train']#[:20000] #80640 samples
    train_labels = pretraining_dict['labels']#[:20000] #80640 samples

    custom_objects = {
        name: obj
        for name, obj in A.__dict__.items()
        if inspect.isclass(obj) and issubclass(obj, tf.keras.layers.Layer)
    }
    custom_objects["LayerScale"] = LayerScale

    encoder = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
    encoder.summary()

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.map(process_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(SSL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    #Extract features
    features_list = []
    labels_list = []

    for images, labels in train_ds:
        features = encoder(images, training=False)
        features = tf.reshape(features, [features.shape[0], -1])
        #Features_list.append(features.numpy())
        #Labels_list.append(labels.numpy())

        features_list.extend(features.numpy().tolist())
        labels_list.extend(labels.numpy().tolist())
        #type1 = type(features_list[0][0])

    #features_array = np.concatenate(features_list, axis=0)
    #labels_array = np.concatenate(labels_list, axis=0)
    features_array = features_list
    labels_array = labels_list

    if use_hdbscan:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        X_normalized = normalize(features_array, norm='l2')

        pca = PCA(n_components=50, random_state=42)
        reduced_embeddings = pca.fit_transform(X_normalized)
        #analyze_embeddings_with_dbscan(reduced_embeddings, labels_array, train_files, class_lookup, output_folder, eps=eps, min_samples=min_samples)
        analyze_embeddings_with_hdbscan(reduced_embeddings, labels_array, train_files, class_lookup, output_folder, min_cluster_size=40, min_samples=40)

    if use_kmeans:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        X_normalized = normalize(features_array, norm='l2')

        pca = PCA(n_components=200, random_state=42)
        reduced_embeddings = pca.fit_transform(X_normalized)
        analyze_embeddings_with_kmeans(reduced_embeddings, labels_array, train_files, class_lookup, output_folder, n_clusters=n_clusters)

        #This is the circular shaped
        pca = PCA(n_components=384, random_state=42)  #n_components=384, only swintranfsformer/comvnexttiny it was 384 because its output feature size is 768
        reduced_embeddings = pca.fit_transform(X_normalized)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)  #TODO
        trans_dim = tsne.fit_transform(np.asarray(reduced_embeddings))
        x = trans_dim[:, 0]
        y = trans_dim[:, 1]
        label_names_list = list(class_lookup.values())
        label_names_list = [label if label != "doppler-ao-descending" else "doppler-ao" for label in label_names_list]
        #plot_tsne(x, y, labels_array, label_names_list, os.path.join(output_folder, 'tsne_from_2048dim_feature_vectors.png'))
        plot_tsne_with_separate_legends(
            x, y,
            labels_array, label_names_list,
            os.path.join(output_folder, 'pca_tsne_from_2048dim_feature_vectors.png'),
            os.path.join(output_folder, 'tsne_legend_vertical.png'),
            os.path.join(output_folder, 'tsne_legend_horizontal.png')
        )

    #Dimensionality reduction with t-SNE
    #tsne = TSNE(n_components=2, perplexity=30, random_state=42) #TODO
    #trans_dim = tsne.fit_transform(np.asarray(features_array))
    #x = trans_dim[:, 0]
    #y = trans_dim[:, 1]
    #label_names_list = list(class_lookup.values())
    #label_names_list = [label if label != "doppler-ao-descending" else "doppler-ao" for label in label_names_list]
    #plot_tsne(x, y, labels_array, label_names_list, os.path.join(output_folder, 'tsne_from_2048dim_feature_vectors.png'))
    #plot_tsne_with_separate_legends(
    #X, y,
    #Labels_array, label_names_list,
    #Os.path.join(output_folder, 'tsne_from_2048dim_feature_vectors.png'),
    #Os.path.join(output_folder, 'tsne_legend_vertical.png'),
    #Os.path.join(output_folder, 'tsne_legend_horizontal.png')
    #)


def find_best_example_per_class(embeddings, labels_array, file_paths, class_lookup, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    labels_array = np.array(labels_array)
    embeddings = np.array(embeddings)
    file_paths = np.array(file_paths)

    best_examples = {}

    for class_id, class_name in class_lookup.items():
        class_id = int(class_id)  #Ensure integer key
        indices = np.where(labels_array == class_id)[0]
        if len(indices) == 0:
            continue

        class_embeddings = embeddings[indices]
        class_center = class_embeddings.mean(axis=0)
        distances = np.linalg.norm(class_embeddings - class_center, axis=1)
        best_idx = indices[np.argmin(distances)]
        best_file = file_paths[best_idx]

        #Save file info
        best_examples[class_name] = {
            "file": best_file,
            "distance_to_class_center": float(distances[np.argmin(distances)]),
            "class_index": class_id,
            "label": class_name
        }

        #Copy and rename
        ext = os.path.splitext(best_file)[1]
        dst_path = os.path.join(output_dir, f"{class_name}{ext}")
        try:
            shutil.copyfile(best_file, dst_path)
            print(f"[OK] Saved best example for {class_name} to {dst_path}")
        except Exception as e:
            print(f"[ERROR] Failed to copy file for {class_name}: {e}")

    return best_examples

def save_training_correction_examples(correction_folder,
                                      mismatch_filename='kmeans_mismatches.json',
                                      output_dir_name='kmeans_corrections_training_data',
                                      json_summary_name='kmeans_summary_counts.json',
                                      in_dict = None):
    """
    Saves example images of training data corrections based on clustering mismatches.

    Args:
        correction_folder (str): Path to the folder containing the mismatch JSON file.
        mismatch_filename (str): Name of the JSON file containing mismatches.
        output_dir_name (str): Subdirectory name to save example correction images.
    """
    mismatch_path = os.path.join(correction_folder, mismatch_filename)
    corrected_training_data = utils.load_json(mismatch_path)

    example_corrections_folder = os.path.join(correction_folder, output_dir_name)
    os.makedirs(example_corrections_folder, exist_ok=True)

    corrections_summary_dict = {}
    max = 10
    cnt=0
    for file, val_dict in corrected_training_data.items():
        if in_dict is not None:
            if file not in in_dict:
                continue

        img = cv2.imread(file)
        if img is None:
            print(f"Warning: Unable to read image {file}")
            continue

        gt_label = val_dict['gt_label']   #If artificial noise has been injected, this wont be the GT label. It will be the noisy label.
        cluster_dominant_label = val_dict['cluster_dominant_label']

        corrections_summary_dict[gt_label] = corrections_summary_dict.get(gt_label, 0) + 1
        index = corrections_summary_dict[gt_label]

        label_dir = os.path.join(example_corrections_folder, gt_label)
        os.makedirs(label_dir, exist_ok=True)

        filename = os.path.basename(file)
        out_filename = f'eg{index}__{cluster_dominant_label}__.png'
        cv2.imwrite(os.path.join(label_dir, out_filename), img)

        cnt+=1
        if cnt == max:
            break

    utils.write_dict_to_json(corrections_summary_dict, correction_folder, json_summary_name)

def get_corrected_labels(correction_folder, train_files, train_labels, class_lookup):#, artificial_noise_file):
    #dict1 = utils.load_json(os.path.join(correction_folder, 'hdbscan_mismatches.json'))
    dict2 = utils.load_json(os.path.join(correction_folder, 'kmeans_mismatches.json'))
    cnt_corrected = 0
    corrected_train_labels = []
    files_needing_correction = {}

    #df = pd.read_csv(artificial_noise_file)
    #noise_dict = df.set_index('file').to_dict(orient='index')

    cnt_equal, cnt_notequal = 0,0
    for i in range(len(train_files)):
        filename = train_files[i]
        corrected_train_labels.append(train_labels[i])

        if filename in dict2:
            val_dict = dict2[filename]
            corrected_train_labels[i] = val_dict['cluster_dominant_class_index']
            cnt_corrected += 1

            files_needing_correction[filename] = {'current_class_index' : train_labels[i],
                                                    'current_class_name' : class_lookup[train_labels[i]],
                                                    'cluster_class_index' : corrected_train_labels[i],
                                                    'cluster_class_name' : class_lookup[corrected_train_labels[i]]}

            #if filename in noise_dict:
            #val2 = noise_dict[filename]
            #if val_dict['cluster_dominant_label'] == val2['original_class']:
            #Cnt_equal+=1
            #else:
            #Cnt_notequal+=1

        #elif filename in dict1:
        #val_dict = dict1[filename]
        #Corrected_train_labels[i] = val_dict['cluster_dominant_class_index']
        #Cnt_corrected += 1

    return corrected_train_labels, files_needing_correction

def remove_matching_entries(file_dict, filenames, labels):
    #Ensure the lists are the same length
    assert len(filenames) == len(labels), "Filenames and labels lists must have the same length."

    #Iterate and keep only those entries that are not in the dictionary
    filtered_filenames = []
    filtered_labels = []

    for filename, label in zip(filenames, labels):
        if os.path.basename(filename) not in file_dict:
            filtered_filenames.append(filename)
            filtered_labels.append(label)

    return filtered_filenames, filtered_labels

def remove_matching_entries2(file_dict, filenames, labels):
    #Ensure the lists are the same length
    assert len(filenames) == len(labels), "Filenames and labels lists must have the same length."

    #Iterate and keep only those entries that are not in the dictionary
    filtered_filenames = []
    filtered_labels = []

    for filename, label in zip(filenames, labels):
        if filename not in file_dict:
            filtered_filenames.append(filename)
            filtered_labels.append(label)

    return filtered_filenames, filtered_labels

def count_matching_mismatches_old(correction_folder, dict3):
    dict1 = utils.load_json(os.path.join(correction_folder, 'hdbscan_mismatches.json'))
    dict2 = utils.load_json(os.path.join(correction_folder, 'kmeans_mismatches.json'))

    count = 0
    for key, val in dict3.items():
        if key in dict1:
            count+=1
        elif key in dict2:
            count+=1

    return count

def count_matching_mismatches(correction_folder, dict3, mode='both'):
    if mode == 'hdbscan':
        dict1 = utils.load_json(os.path.join(correction_folder, 'hdbscan_mismatches.json'))
    elif mode == 'kmeans':
        dict2 = utils.load_json(os.path.join(correction_folder, 'kmeans_mismatches.json'))
    elif mode == 'both':
        dict1 = utils.load_json(os.path.join(correction_folder, 'hdbscan_mismatches.json'))
        dict2 = utils.load_json(os.path.join(correction_folder, 'kmeans_mismatches.json'))

    count = 0
    for key in dict3:
        if mode == 'hdbscan' and key in dict1:
            count += 1
        elif mode == 'kmeans' and key in dict2:
            count += 1
        elif mode == 'both' and (key in dict1 or key in dict2):
            count += 1
    return count

def compute_sensitivity_specificity(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    num_classes = cm.shape[0]

    sensitivity = []
    specificity = []

    total_TP = 0
    total_FN = 0
    total_FP = 0
    total_TN = 0

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        total_TP += TP
        total_FN += FN
        total_FP += FP
        total_TN += TN

        sensitivity.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)

    #Convert to arrays
    sensitivity = np.array(sensitivity)
    specificity = np.array(specificity)

    #Macro averages (unweighted mean)
    macro_sens = np.mean(sensitivity)
    macro_spec = np.mean(specificity)

    #Micro averages (global TP, FN, FP, TN)
    micro_sens = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    micro_spec = total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0

    return {
        "per_class": {
            "sensitivity": sensitivity,
            "specificity": specificity
        },
        "macro_avg": {
            "sensitivity": macro_sens,
            "specificity": macro_spec
        },
        "micro_avg": {
            "sensitivity": micro_sens,
            "specificity": micro_spec
        }
    }

def get_files_labels_from_cat_set(reversed_class_lookup):
    #Define file paths and dictionary keys
    file_paths = [
        'view_classification_tests_with_catherine_labels__agree_with_darrel.json',
        'view_classification_tests_with_catherine_labels__disagree_with_both.json',
        'view_classification_tests_with_catherine_labels__none.json',
        'view_classification_tests_with_catherine_labels__notsure.json',
        'view_classification_tests_with_catherine_labels__agree_with_ai.json'
    ]

    files, labels = [], []

    for file_path in file_paths:
        data_dict = utils.load_json(file_path)
        for key, value in data_dict.items():
            files.append(key)
            labels.append(reversed_class_lookup[value['GT']])

    return files, labels

def compute_epoch_correction_statistics(
    epoch,
    correction_folder,
    epoch_correction_folder,
    cat_agree_with_darrel_dict,
    cat_disagree_dict,
    cat_none_dict,
    dual_dict,
    train_dict,
    val_dict,
    filename,
    reversed_class_lookup,
    mode='both',
    artificial_noise_file=None
):
    cnt_cat_agree_with_darrel_corr = count_matching_mismatches(epoch_correction_folder, cat_agree_with_darrel_dict, mode)
    cnt_cat_disagree_corr = count_matching_mismatches(epoch_correction_folder, cat_disagree_dict, mode)
    cnt_cat_none_corr = count_matching_mismatches(epoch_correction_folder, cat_none_dict, mode)

    total_cat_agree_with_darrel_no_duel = len(cat_agree_with_darrel_dict) - len(set(dual_dict) & set(cat_agree_with_darrel_dict))
    total_cat_disagree_no_duel = len(cat_disagree_dict) - len(set(dual_dict) & set(cat_disagree_dict))
    total_cat_none_no_duel = len(cat_none_dict) - len(set(dual_dict) & set(cat_none_dict))

    perc_cat_agree_with_darrel = (cnt_cat_agree_with_darrel_corr / total_cat_agree_with_darrel_no_duel) * 100.0 if total_cat_agree_with_darrel_no_duel > 0 else 0
    perc_cnt_cat_disagree_corr = (cnt_cat_disagree_corr / total_cat_disagree_no_duel) * 100.0 if total_cat_disagree_no_duel > 0 else 0
    perc_cnt_cat_none_corr = (cnt_cat_none_corr / total_cat_none_no_duel) * 100.0 if total_cat_none_no_duel > 0 else 0

    cnt_train_corr = count_matching_mismatches(epoch_correction_folder, train_dict, mode)
    cnt_val_corr = count_matching_mismatches(epoch_correction_folder, val_dict, mode)

    percentage_train_noise_detected_of_artificial_noise = 0
    percentage_train_noise_detected_of_total_noise = 0
    cnt_actual_noise_detected_of_artificial_noise = 0
    combined_score = 0
    detection_rate = 0
    cluster_accuracy_on_detected_artificial_noise_avg, sensitivity_avg, specificity_avg, precision_avg, f1_avg = 0,0,0,0,0
    cnt_accurate_detected_noise_of_artificial_noise = 0
    detection_accuracy, correction_accuracy = 0, 0
    synthentic_noise_count = 0
    if os.path.exists(artificial_noise_file):
        df = pd.read_csv(artificial_noise_file)
        noise_dict = df.set_index('file').to_dict(orient='index')
        synthentic_noise_count = len(noise_dict)
        cnt_actual_noise_detected_of_artificial_noise = count_matching_mismatches(epoch_correction_folder, noise_dict, mode)
        detection_rate = cnt_actual_noise_detected_of_artificial_noise / len(noise_dict)
        percentage_train_noise_detected_of_artificial_noise = detection_rate * 100.0
        percentage_train_noise_detected_of_total_noise = (cnt_actual_noise_detected_of_artificial_noise / cnt_train_corr)*100.0
        print(f'train_noise_to_artificial_noise:{percentage_train_noise_detected_of_artificial_noise}%')
        print(f'train_noise_to_total_noise:{percentage_train_noise_detected_of_total_noise}%')

        #Also check how much of the artificial noise ended up in the correct cluster:
        if mode == 'kmeans':
            kmeans_dict = utils.load_json(os.path.join(epoch_correction_folder, 'kmeans_mismatches.json'))
        if  mode == 'hdbscan':
            hdbscan_dict = utils.load_json(os.path.join(epoch_correction_folder, 'hdbscan_mismatches.json'))

        labels_gt, labels_cluster = [], []
        for file, val in noise_dict.items():
            gt_label = reversed_class_lookup[val['original_class']]
            if mode == 'kmeans' and file in kmeans_dict:
                cluster_val = kmeans_dict[file]
                cluster_dominant_label = cluster_val['cluster_dominant_class_index']
                labels_cluster.append(cluster_dominant_label)
                labels_gt.append(gt_label)
            elif mode == 'hdbscan' and file in hdbscan_dict:
                cluster_val = hdbscan_dict[file]
                cluster_dominant_label = cluster_val['cluster_dominant_class_index']
                labels_cluster.append(cluster_dominant_label)
                labels_gt.append(gt_label)
            elif mode == 'both':
                if file in kmeans_dict:
                    cluster_val = kmeans_dict[file]
                    cluster_dominant_label = cluster_val['cluster_dominant_class_index']
                    labels_cluster.append(cluster_dominant_label)
                    labels_gt.append(gt_label)
                elif file in hdbscan_dict:
                    cluster_val = hdbscan_dict[file]
                    cluster_dominant_label = cluster_val['cluster_dominant_class_index']
                    labels_cluster.append(cluster_dominant_label)
                    labels_gt.append(gt_label)

            #else:
                #Not detected as noise but is part of the artificial noise
                #Then it was trained to belong to the new class label
                #new_class_label = reversed_class_lookup[val['new_class']]
                #Labels_cluster.append(new_class_label)
                #Labels_gt.append(gt_label)
        cluster_accuracy_on_detected_artificial_noise_avg = accuracy_score(labels_gt, labels_cluster)
        precision, recall, f1score, support = score(labels_gt, labels_cluster)
        result_metrics = compute_sensitivity_specificity(labels_gt, labels_cluster)
        sensitivity_avg = result_metrics["macro_avg"]["sensitivity"]
        specificity_avg = result_metrics["macro_avg"]["specificity"]
        precision_avg = np.mean(precision)
        #recall_avg = np.mean(recall) not needed here, same as sensitivity
        f1_avg = np.mean(f1score)
        combined_score = detection_rate * cluster_accuracy_on_detected_artificial_noise_avg
        cnt_accurate_detected_noise_of_artificial_noise = cnt_actual_noise_detected_of_artificial_noise * cluster_accuracy_on_detected_artificial_noise_avg
        detection_accuracy = (cnt_actual_noise_detected_of_artificial_noise / len(noise_dict)) * 100.0          #Same as detection rate
        label_recovery_precision = (cnt_accurate_detected_noise_of_artificial_noise / len(noise_dict)) * 100.0  #Same as lrp
        correction_accuracy = (cnt_accurate_detected_noise_of_artificial_noise / cnt_actual_noise_detected_of_artificial_noise) * 100.0

    results_dict = {
        "epoch": epoch,
        "synthentic_noise_count": synthentic_noise_count,
        "train_corr": cnt_train_corr,
        "val_corr": cnt_val_corr,
        "cnt_cat_agree_with_darrel_corr": cnt_cat_agree_with_darrel_corr,
        "total_cat_agree_with_darrel_no_duel": total_cat_agree_with_darrel_no_duel,
        "perc_cat_agree_with_darrel": perc_cat_agree_with_darrel,
        "cnt_cat_disagree_corr": cnt_cat_disagree_corr,
        "total_cat_disagree_no_duel": total_cat_disagree_no_duel,
        "perc_cnt_cat_disagree_corr": perc_cnt_cat_disagree_corr,
        "cnt_cat_none_corr": cnt_cat_none_corr,
        "total_cat_none_no_duel": total_cat_none_no_duel,
        "perc_cnt_cat_none_corr": perc_cnt_cat_none_corr,
        "cnt_actual_noise_detected_of_artificial_noise":cnt_actual_noise_detected_of_artificial_noise,
        "percentage_actual_noise": percentage_train_noise_detected_of_artificial_noise,
        "percentage_actual_to_total_noise": percentage_train_noise_detected_of_total_noise,
        "accuracy": cluster_accuracy_on_detected_artificial_noise_avg,
        "sensitivity": sensitivity_avg,
        "specificity": specificity_avg,
        "precision": precision_avg,
        "f1": f1_avg,
        "combined_score": combined_score,
        "train_corr1": cnt_train_corr,                                                                       #N(hat)
        "cnt_actual_noise_detected_of_artificial_noise1": cnt_actual_noise_detected_of_artificial_noise,     #N(hat) intersection N
        "cnt_accurate_detected_noise_of_artificial_noise": cnt_accurate_detected_noise_of_artificial_noise,  #C
        "detection_accuracy": detection_accuracy,  #DR
        "label_recovery_precision": label_recovery_precision,  #LRP
        "correction_accuracy": correction_accuracy,
    }

    utils.write_dict_to_json(results_dict, correction_folder, filename)
    return results_dict

def run_self_correction_pipeline(train_files, train_labels,
                                  val_files, val_labels,
                                  reversed_class_lookup,
                                  class_lookup,
                                  save_folder,
                                  ssl_folder,
                                  correction_folder,
                                  dual_dict,
                                  artificial_noise_file):
    epochs = [0]#[0, 5, 10, 15, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    #Combine files for self-correction (not for training)
    combined_files = train_files #+ val_files
    combined_labels = train_labels #+ val_labels

    hard_files, hard_labels = get_files_labels_from_cat_set(reversed_class_lookup)
    #Combined_files.extend(hard_files)
    #Combined_labels.extend(hard_labels)

    combined_data = {'train': combined_files, 'labels': combined_labels}
    utils.write_dict_to_json(combined_data, save_folder, 'combined_files_labels.json')

    combined_file = os.path.join(save_folder, 'combined_files_labels.json')

    for epoch in epochs:
        model_file = os.path.join(ssl_folder, f'model_ep_{epoch}.h5')

        if epoch == 0:
            model_file = os.path.join(ssl_folder, f'best_model.h5')

        if not os.path.exists(model_file): continue

        epoch_correction_folder = correction_folder + f'_{epoch}'
        os.makedirs(epoch_correction_folder, exist_ok=True)

        extract_features_and_cluster_directly(combined_file, model_file, epoch_correction_folder, class_lookup) #TODO

        train_dict = {k: 1 for k in train_files}
        val_dict = {k: 1 for k in val_files}

        cat_agree_with_darrel_dict = utils.load_json('view_classification_tests_with_catherine_labels__agree_with_darrel.json')
        cat_disagree_dict = utils.load_json('view_classification_tests_with_catherine_labels__disagree_with_both.json')
        cat_none_dict = utils.load_json('view_classification_tests_with_catherine_labels__none.json')

        for mode in ['kmeans']:#['both', 'hdbscan', 'kmeans']:
            filename = f"correction_statistics_{epoch}_{mode}.json"
            compute_epoch_correction_statistics(
                epoch=epoch,
                correction_folder=correction_folder,
                epoch_correction_folder=epoch_correction_folder,
                cat_agree_with_darrel_dict=cat_agree_with_darrel_dict,
                cat_disagree_dict=cat_disagree_dict,
                cat_none_dict=cat_none_dict,
                dual_dict=dual_dict,
                train_dict=train_dict,
                val_dict=val_dict,
                filename=filename,
                reversed_class_lookup = reversed_class_lookup,
                mode=mode,
                artificial_noise_file = artificial_noise_file
            )

        #Filter and save mismatches for training files only
        kmeans_dict = utils.load_json(os.path.join(epoch_correction_folder, 'kmeans_mismatches.json'))
        kmeans_mismatches_train = {k: v for k, v in kmeans_dict.items() if k in train_dict}
        utils.write_dict_to_json(kmeans_mismatches_train, epoch_correction_folder, "kmeans_mismatches_train.json")

        #Save correction examples
        for split_name, data_dict in {
            'training': train_dict,
            'val': val_dict,
            'cat_disagree': cat_disagree_dict
        }.items():
            for method in ['kmeans']:#['hdbscan', 'kmeans']:
                save_training_correction_examples(
                    epoch_correction_folder,
                    f'{method}_mismatches.json',
                    f'{method}_corrections_{split_name}_data',
                    f'{method}_summary_counts.json',
                    data_dict
                )

def scale_epochs_for_plotting(epochs, best_epoch, split_point=20, small_step=2, large_step=2.5):
    """
    Scale early epochs with smaller spacing and later epochs with larger spacing,
    excluding best_epoch from position mapping.
    """
    scaled_epochs = {}
    current_pos = 0
    for epoch in sorted(epochs):
        if epoch == best_epoch:
            continue  #Exclude from spacing logic
        scaled_epochs[epoch] = current_pos
        if epoch < split_point:
            current_pos += small_step
        else:
            current_pos += large_step
    #Add best_epoch manually just for plotting the red dot
    if best_epoch in epochs:
        #Estimate its position based on surrounding epochs
        sorted_epochs = sorted(e for e in epochs if e != best_epoch)
        idx = sorted_epochs.index(min([e for e in sorted_epochs if e > best_epoch], default=sorted_epochs[-1]))
        if idx == 0:
            pos = scaled_epochs[sorted_epochs[0]] - small_step
        else:
            prev_ep = sorted_epochs[idx - 1]
            next_ep = sorted_epochs[idx]
            pos = (scaled_epochs[prev_ep] + scaled_epochs[next_ep]) / 2
        scaled_epochs[best_epoch] = pos
    return scaled_epochs


def plot_metric(ax, epochs, data_dict, label, color, best_epoch, scaled_positions):
    x_vals = [scaled_positions[e] for e in epochs]
    y_vals = [data_dict[e] for e in epochs]

    ax.plot(x_vals, y_vals, label=label, color=color,
            linewidth=2.5, marker='o', markersize=8)

    #Highlight best_epoch's marker in red if it's in the data
    if best_epoch in data_dict and best_epoch in scaled_positions:
        ax.plot(scaled_positions[best_epoch], data_dict[best_epoch],
                marker='o', color='red', markersize=8, linestyle='None')


def tidy_plot(ax, title, ylabel, xticks, xtick_labels, show_legend = True):
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.tick_params(axis='both', width=2, length=6, labelsize=13)
    ax.grid(True, linestyle='--', alpha=0.7)
    if show_legend:
        ax.legend(title="Method", fontsize=13, title_fontsize=14)

def plot_correction_statistics(correction_folder):
    methods = ['kmeans', 'hdbscan']
    metric_keys = {
        "disagree_data": "perc_cnt_cat_disagree_corr",
        "train_data": "train_corr",
        "noise_to_actual_noise": "percentage_actual_noise",
        "noise_to_total_noise": "percentage_actual_to_total_noise",
        "accuracy": "accuracy",
        "combined_score": "combined_score",
        "sensitivity": "sensitivity",
        "specificity": "specificity",
        "precision": "precision",
        "actual_noise_detected_of_artificial_noise": 'cnt_actual_noise_detected_of_artificial_noise',
        "accurate_detected_noise_of_artificial_noise": 'cnt_accurate_detected_noise_of_artificial_noise',
        'detection_accuracy': 'detection_accuracy',
        'correction_accuracy': 'correction_accuracy',
    }
    colors = {
        'both': 'orchid',
        'kmeans': 'cornflowerblue',
        'hdbscan': 'forestgreen'
    }

    data = {k: {m: {} for m in methods} for k in metric_keys}
    base_path = os.path.dirname(correction_folder)
    df = pd.read_csv(os.path.join(base_path, 'ssl/epoch_history.csv'))
    best_epoch = df['val_loss'].idxmin() + 1
    #epochs_to_plot = sorted(set([5, 10, 15, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, best_epoch]))
    epochs_to_plot = sorted(set([5, 10, 15, 20, 40, 60, 80, 100, 120, best_epoch]))
    available_epochs = set()

    for method in methods:
        for epoch in epochs_to_plot:
            fname = f"correction_statistics_{0 if epoch == best_epoch else epoch}_{method}.json"
            fpath = os.path.join(correction_folder, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r') as f:
                stats = json.load(f)
            for key, val in metric_keys.items():
                data[key][method][epoch] = stats[val]
            available_epochs.add(epoch)

    scaled_positions = scale_epochs_for_plotting(available_epochs, best_epoch)
    #xtick_epochs = sorted(e for e in available_epochs if e != best_epoch)
    #xticks = [scaled_positions[e] for e in xtick_epochs]
    #xtick_labels = [str(e) for e in xtick_epochs]
    xtick_epochs = sorted(available_epochs)
    xticks = [scaled_positions[e] for e in xtick_epochs]
    xtick_labels = [str(e) for e in xtick_epochs]

    plots = [
        ("disagree_data", "Correction Rate in Disagree Cases", "Percentage Corrected", "plot_correction_disagree_percentage.png"),
        ("train_data", None, "Detection Count", "plot_total_mismatches_in_train.png"),
        ("noise_to_actual_noise", "Ratio of Detected Noise to Artificial Noise", "Detected Noise Ratio (%)", "plot_ratio_detected_noise_to_artificial_noise.png"),
        ("noise_to_total_noise", "Ratio of Detected Noise to Total Noise", "Detected Noise Ratio (%)", "plot_ratio_detected_noise_to_all_noise.png"),
        ("accuracy", "Accuracy of Detection", "Detection Accuracy (%)", "plot_accuracy_on_detected_artificial_noise.png"),
        ("combined_score", "Combined Score (Coverage x Accuracy)", "Score", "plot_combined_score.png"),
        ("sensitivity", "Sensitivity", "Sensitivity", "plot_sensitivity.png"),
        ("specificity", "Specificity", "Specificity", "plot_specificity.png"),
        ("precision", "Precision", "Precision", "plot_precision.png"),
        ("actual_noise_detected_of_artificial_noise", "Detected Noise (#)", "Detected Noise", "plot_actual_noise_detected_of_artificial_noise.png"),
        ("accurate_detected_noise_of_artificial_noise", "Detected Noise (#)", "Accurately Detected Noise", "plot_accurate_detected_noise_of_artificial_noise.png"),
    ]

    for metric_key, title, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        for method in methods:
            epochs = sorted(data[metric_key][method].keys())
            plot_metric(ax, epochs, data[metric_key][method], method, colors[method], best_epoch, scaled_positions)
        tidy_plot(ax, title, ylabel, xticks, xtick_labels)
        fig.tight_layout()
        fig.savefig(os.path.join(correction_folder, filename), bbox_inches='tight', dpi=300)
        plt.close(fig)

    #Plot kmeans only variant for each of the above graphs
    for metric_key, title, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        for method in methods[0:1]:
            epochs = sorted(data[metric_key][method].keys())
            plot_metric(ax, epochs, data[metric_key][method], method, 'goldenrod', best_epoch, scaled_positions)
        tidy_plot(ax, title, ylabel, xticks, xtick_labels, show_legend=False)
        fig.tight_layout()
        filename = filename[0:filename.rindex('.')] + '_kmeans.png'
        fig.savefig(os.path.join(correction_folder, filename), bbox_inches='tight', dpi=300)
        plt.close(fig)

    #Additional plot: both noise detection metrics on one plot, only for kmeans
    joint_metric_keys = [
        ("actual_noise_detected_of_artificial_noise", "Total Detected Noise of Artificial Noise", "cornflowerblue"),
        ("accurate_detected_noise_of_artificial_noise", "Accurately Corrected Noise of Artificial Noise", "mediumseagreen"),
        ("train_data", "All Noise", "goldenrod")
    ]
    joint_plots = [
        {
            "metrics": [
                ("actual_noise_detected_of_artificial_noise", "Total Detected Noise of Artificial Noise", "cornflowerblue"),
                ("accurate_detected_noise_of_artificial_noise", "Accurately Corrected Noise of Artificial Noise", "mediumseagreen"),
                ("train_data", "All Noise", "goldenrod")
            ],
            "filename": "plot_kmeans_noise_detection_joint",
            "ylabel": "Detection Count"
        },
        {
            "metrics": [
                ("detection_accuracy", "Dectection Accuracy", "cornflowerblue"),
                ("correction_accuracy", "Correction Accuracy", "mediumseagreen"),
            ],
            "filename": "plot_kmeans_detection_accuracy_vs_correction_accuracy",
            "ylabel": "Accuracy (%)"
        }
    ]

    for joint in joint_plots:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        plot_joint_metrics(
            ax=ax,
            joint_metrics=joint["metrics"],
            data=data,
            method='kmeans',
            best_epoch=best_epoch,
            scaled_positions=scaled_positions,
            xticks=xticks,
            xtick_labels=xtick_labels,
            ylabel=joint["ylabel"]
        )

        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                           ncol=2, fontsize=13, title_fontsize=14, frameon=False,
                           handlelength=3.0, handleheight=2.0, markerscale=1.6)

        #Save legend separately
        legend_fig = plt.figure(figsize=(4, 1), dpi=300)
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis('off')
        legend_for_save = legend_ax.legend(*ax.get_legend_handles_labels(),
                                           loc='center', ncol=2, frameon=False, fontsize=13, title_fontsize=14,
                                           handlelength=3.0, handleheight=2.0, markerscale=1.6)
        legend_fig.tight_layout()
        legend_fig.savefig(os.path.join(correction_folder, f"legend_{joint['filename']}.png"),
                           bbox_inches='tight', dpi=300)
        plt.close(legend_fig)

        #Save plot without legend
        ax.legend_.remove()
        fig.tight_layout()
        fig.savefig(os.path.join(correction_folder, f"{joint['filename']}_no_legend.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)


def plot_joint_metrics(ax, joint_metrics, data, method, best_epoch, scaled_positions,
                       xticks, xtick_labels, ylabel, title=None):
    for key, label, color in joint_metrics:
        epochs = sorted(data[key][method].keys())
        x_vals = [scaled_positions[e] for e in epochs]
        y_vals = [data[key][method][e] for e in epochs]

        ax.plot(x_vals, y_vals, label=label, color=color,
                linewidth=2.5, marker='o', markersize=8)

        #Highlight best_epoch's marker in red
        if best_epoch in data[key][method]:
            ax.plot(scaled_positions[best_epoch], data[key][method][best_epoch],
                    marker='o', color='red', markersize=8, linestyle='None')

    tidy_plot(ax, title, ylabel, xticks, xtick_labels)


def inject_label_noise(train_files: List[str],
                       train_labels: List[int],
                       class_lookup: Dict[str, str],
                       groupings_dict: Dict[str, Dict[str, int]],
                       noise_percentages: Dict[str, float]
                      ) -> Tuple[List[int], List[Dict[str, str]]]:
    """
    Injects label noise into train_labels by reassigning a percentage of labels
    within the same group to simulate mislabeling, and logs the changes.

    Args:
        train_files: List of filenames.
        train_labels: List of integer labels corresponding to class indices.
        class_lookup: Dictionary mapping index (as string) to class name.
        groupings_dict: Groupings of class names by anatomical region.
        noise_percentages: Dictionary of noise percentages per group, e.g., {"apical": 1, "plax": 2}

    Returns:
        Tuple of (new train_labels list, list of changes as dicts)
    """
    label_to_class = {int(k): v for k, v in class_lookup.items()}
    class_to_label = {v: k for k, v in label_to_class.items()}

    #Map class names to group
    class_to_group = {}
    for group, members in groupings_dict.items():
        for cls in members:
            class_to_group[cls] = group

    #Organize indices of labels by group
    group_indices = defaultdict(list)
    for idx, label in enumerate(train_labels):
        class_name = label_to_class[label]
        group = class_to_group.get(class_name)
        if group:
            group_indices[group].append(idx)

    new_labels = train_labels.copy()
    change_log = []

    for group, indices in group_indices.items():
        noise_pct = noise_percentages.get(group, 0)
        noise_pct/=100.0
        num_noisy = int(len(indices) * noise_pct)
        noisy_indices = random.sample(indices, num_noisy)

        group_classes = list(groupings_dict[group].keys())
        group_labels = [int(class_to_label[cls]) for cls in group_classes]

        for idx in noisy_indices:
            current_label = new_labels[idx]
            possible_labels = [lbl for lbl in group_labels if lbl != current_label]
            if possible_labels:
                new_label = random.choice(possible_labels)
                new_labels[idx] = new_label

                change_log.append({
                    "file": train_files[idx],
                    "original_class": label_to_class[current_label],
                    "new_class": label_to_class[new_label],
                    "group": group
                })

    return new_labels, change_log

def inject_from_noise_file(iteration_folder, train_files, train_labels, reversed_class_lookup):
    train_labels_with_noise = train_labels.copy()
    if os.path.exists(os.path.join(iteration_folder, "injected_label_noise_log.csv")):
        df = pd.read_csv(os.path.join(iteration_folder, "injected_label_noise_log.csv"))
        data_dict = df.set_index('file').to_dict(orient='index')
        cnt_noise_in_train = 0
        for i in range(len(train_files)):
            if train_files[i] in data_dict:
                val = data_dict[train_files[i]]
                noise_label = reversed_class_lookup[val['new_class']]
                curr_label = train_labels_with_noise[i]
                train_labels_with_noise[i] = noise_label
                cnt_noise_in_train += 1
        print(f'Noise injected into train: {cnt_noise_in_train}')
    return train_labels_with_noise

#Main orchestrator - wires together setup, training modes, and evaluations.
def main():
    #Tf.config.run_functions_eagerly(True) #uncomment when debugging

    #print(f'The current precision policy is: {tf.keras.mixed_precision.global_policy()}')
    #Tf.keras.mixed_precision.set_global_policy('mixed_float16')
    #print(f'The current precision policy is: {tf.keras.mixed_precision.global_policy()}')

    seed = 444
    random.seed(seed)

    #if len(sys.argv) > 1:
    #seed = int(sys.argv[1])

    print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS", "Not set"))
    print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS", "Not set"))
    print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS", "Not set"))

    parser = argparse.ArgumentParser(description="Echocardiography Training Pipeline")

    #Core experiment settings
    parser.add_argument("--pretraining", action='store_true', help="Enable pretraining")
    parser.add_argument("--correction", action='store_true', help="Enable correction phase")
    parser.add_argument("--downstream_with_imagenet", action='store_true',
                        help="Use ImageNet weights for downstream task")
    parser.add_argument("--downstream_with_rand_init", action='store_true',
                        help="Use random weights for downstream task")
    parser.add_argument("--downstream_training", action='store_true', help="Train downstream classifier")
    parser.add_argument("--downstream_with_pretrainedweights", action='store_true',
                        help="Use contrastive pretrained weights")
    parser.add_argument("--downstream_with_linear", action='store_true',
                        help="Use contrastive pretrained weights but lock encoder")

    #Noise correction options
    parser.add_argument("--use_corrections", action='store_true', help="Use correction labels in training")
    parser.add_argument("--remove_as_noisy_labels", action='store_true',
                        help="Remove noisy labels instead of correcting")
    parser.add_argument("--use_epoch_weights", type=int, default=25,
                        help="Epoch at which weights are selected for fine-tuning")
    parser.add_argument("--noise_percentage", type=int, default=0, help="Percentage of label noise to inject")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature of loss")
    parser.add_argument("--drop_k", type=int, default=0, help="Drop K positives in DropCon")
    parser.add_argument("--backbone", type=int, default=0, help="Choice of Backbone")
    parser.add_argument("--loss", type=int, default=0, help="Choice of loss function")

    args = parser.parse_args()

    #You can now access variables like:
    pretraining = args.pretraining
    correction = args.correction
    downstream_with_imagenet = args.downstream_with_imagenet
    downstream_with_random_init = args.downstream_with_rand_init
    downstream_training = args.downstream_training
    downstream_with_pretrainedweights = args.downstream_with_pretrainedweights
    downstream_with_linear = args.downstream_with_linear
    use_corrections = args.use_corrections
    remove_as_noisy_labels = args.remove_as_noisy_labels
    use_epoch_weights = args.use_epoch_weights
    noise_percentage = args.noise_percentage
    if args.temperature!=0:
        global TEMPERATURE
        TEMPERATURE = args.temperature
    if args.drop_k!=0:
        global DROP_K
        DROP_K = args.drop_k

    global USE_BACKBONE
    USE_BACKBONE = BACKBONES[args.backbone]

    global USE_LOSS
    USE_LOSS = LOSSES[args.loss]

    #CHANGE RUN SETTINGS HERE:
    augmentation = True
    train_percentage = 84
    split_sets = False

    if pretraining and (USE_BACKBONE == BACKBONES[10] or USE_BACKBONE == BACKBONES[8]):
        print(f'The current precision policy is: {tf.keras.mixed_precision.global_policy()}')
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f'The current precision policy is: {tf.keras.mixed_precision.global_policy()}')

    #Manual
    #pretraining = True
    #correction = False
    #downstream_with_random_init = False
    #downstream_with_imagenet = False
    #downstream_with_pretrainedweights = True
    #downstream_with_linear = False
    #downstream_training = False #if false, dont train, only run inference.
    #use_corrections = False #this or remove_as_noisy_labels:
    #remove_as_noisy_labels = False
    #use_epoch_weights = 0
    #noise_percentage = 20 #0 downstream_with_imagenet-no correction, 50 downstream_with_imagenet-no correction-use_corrections

    #Global USE_LOSS
    #USE_LOSS = LOSSES[1]
    #inject_noise = False #Inject artificial label noise
    if noise_percentage>0:
        inject_noise = True
    noise_percentages = {"apical": noise_percentage, "plax": noise_percentage, "psax": noise_percentage, "doppler": noise_percentage, "mmode": noise_percentage}
    noise_tag = "_".join([
        f"{k[:2]}{v}" if k != "doppler" else f"do{v}"
        for k, v in noise_percentages.items()
    ])

    data_folder = r'Data/Unity-Classification-B'
    save_folder = f'/mnt/ssd/preshen/backup/run_view_with_noise_Aug-{augmentation}_T{str(TEMPERATURE).replace(".", "")}_{USE_LOSS}_{USE_BACKBONE}'
    if USE_LOSS == LOSSES[1]:
          save_folder = save_folder + f'_k{DROP_K}_v{DROP_V}'
    save_folder = save_folder + '_run_0'

    iteration_folder = os.path.join(save_folder, f'{noise_tag}')
    info_folder = 'info' #Os.path.join(save_folder, 'info')
    downstream_with_imagenet_folder = os.path.join(iteration_folder, 'downstream_with_imagenet')
    downstream_with_random_init_folder = os.path.join(iteration_folder, 'downstream_with_random_init')
    downstream_with_pretrained_folder = os.path.join(iteration_folder, 'downstream_with_pretrained_model')
    downstream_with_linear_folder = os.path.join(iteration_folder, 'downstream_with_linear')
    if use_corrections:
        downstream_with_imagenet_folder = downstream_with_imagenet_folder + f'_with_corrected_labels_{use_epoch_weights}'
        downstream_with_pretrained_folder = downstream_with_pretrained_folder + f'_with_corrected_labels_{use_epoch_weights}'
    elif remove_as_noisy_labels:
        downstream_with_imagenet_folder = downstream_with_imagenet_folder + f'_with_removed_noisy_labels_{use_epoch_weights}'
        downstream_with_pretrained_folder = downstream_with_pretrained_folder + f'_with_removed_noisy_labels_{use_epoch_weights}'
    else:
        downstream_with_pretrained_folder = downstream_with_pretrained_folder + f'_{use_epoch_weights}'
        downstream_with_linear_folder = downstream_with_linear_folder + f'_{use_epoch_weights}'

    ssl_folder = os.path.join(iteration_folder, 'ssl')
    correction_folder = os.path.join(iteration_folder, 'correction')
    for folder in [downstream_with_imagenet_folder, ssl_folder, correction_folder, downstream_with_pretrained_folder]:
        os.makedirs(folder, exist_ok=True)

    #Files_for_relabelling are the ones selected to be relabelled. Exclude these files for now
    #But add them back when the new labels are available(add them to the training set,
    #So the model can be trained with correct labels).
    files_for_relabelling = r'view classification files for relabelling.txt'
    files_for_relabelling_dict={}
    with open(files_for_relabelling) as file:
        for line in file:
            files_for_relabelling_dict[line.rstrip()]=1

    #List of files contain dual fan-shaped scans
    dual_files_csv = 'dual_image_predictions.csv'
    data = pd.read_csv(dual_files_csv)
    dual_dict = {f'Data/Unity-Classification-B/{value.strip()}': idx for idx, value in data['filename'].items()}

    #BEGIN EXPERIMENT**************

    random.seed(444)
    if split_sets: #Should only run once
        (train_files, train_labels,
         val_files, val_labels,
         test_files, test_labels,
         class_count, class_lookup,
         class_count_train, class_count_val, class_count_test,
         num_excluded_dual_per_class) = split_data(data_folder=data_folder,
                                                   perc_train=train_percentage / 100.0,
                                                   perc_val=0.10,
                                                   perc_test=0.06,
                                                   seed=seed,
                                                   files_for_relabelling=files_for_relabelling_dict, dual_files_dict=dual_dict)
                                                    #files_for_relabelling=files_for_relabelling_dict)
        reversed_class_lookup = {v: k for k, v in class_lookup.items()}

        total_dual_found = sum(value for value in num_excluded_dual_per_class.values())
        num_excluded_dual_per_class['total'] = total_dual_found
        utils.write_dict_to_json(num_excluded_dual_per_class, iteration_folder, 'excluded_dual_per_class.json')

        utils.write_dict_to_json({'train_files': train_files, 'train_labels': train_labels,
                                  'val_files': val_files, 'val_labels': val_labels,
                                  'test_files': test_files, 'test_labels': test_labels},
                                 iteration_folder, 'data_split.json')

        utils.write_dict_to_json(class_lookup,
                                 iteration_folder, 'class_lookup.json')
        utils.write_dict_to_json(class_count,
                                 iteration_folder, 'class_count.json')
        utils.write_dict_to_json(class_count_train,
                                 iteration_folder, 'class_count_train.json')
        utils.write_dict_to_json(class_count_val,
                                 iteration_folder, 'class_count_val.json')
        utils.write_dict_to_json(class_count_test,
                                 iteration_folder, 'class_count_test.json')
    else:
        class_lookup = utils.load_json(os.path.join(info_folder, 'class_lookup.json'))
        class_lookup = {int(k): v for k, v in class_lookup.items()}
        reversed_class_lookup = {v: k for k, v in class_lookup.items()}
        data_split_dict = utils.load_json(os.path.join(info_folder, 'data_split.json'))
        train_files = data_split_dict['train_files']
        train_labels = data_split_dict['train_labels']
        val_files = data_split_dict['val_files']
        val_labels = data_split_dict['val_labels']
        test_files = data_split_dict['test_files']
        test_labels = data_split_dict['test_labels']
        unique, counts = np.unique(train_labels, return_counts=True)
        unique1, counts1 = np.unique(val_labels, return_counts=True)
        unique2, counts2 = np.unique(test_labels, return_counts=True)
        #Should all be equal:
        print(f'training set: {len(train_files)}')
        print(f'val set: {len(val_files)}')
        print(f'test set: {len(test_files)}')
        print(f'Number of classes in training set: {len(unique)}')
        print(f'Number of classes in val set: {len(unique1)}')
        print(f'Number of classes in test set: {len(unique2)}')

    #The three lists below are just the labels for each expert separately, also corresponds to majority_vote_files
    #Counts are 4894 for all three, but may contain None values for expert 2 and expert 3.
    #Gt_vote_labels, majority_vote_files, files with majority vote and its labels (best 2 out of 3), count = 4894
    #All_agree_files, files where all three experts agree on the label, count = 3327
    #No_maj_vote_files are the files where all experts disagree. expkappa_other_labels are there corresponding labels. count = 553
    #4894 + 553 will match the count of the full test set, i.e. test_files
    (exp1_labels, exp2_labels, exp3_labels,
     gt_vote_labels, majority_vote_files,
     all_agree_files,
     no_maj_vote_files, exp1_other_labels, exp2_other_labels, exp3_other_labels) = get_expert_labels(test_files, test_labels, class_lookup,
                                                                                                     reversed_class_lookup,
                                                                                                     save_dir="view_classification_label_analysis_exp1",
                                                                                                     fallback_strategy='expert1',
                                                                                                     create_pdf = False)

    #All evaluation Sets
    filtered_idxs = [
        i for i in range(len(gt_vote_labels))
        if exp1_labels[i] is not None and exp2_labels[i] is not None and exp3_labels[i] is not None
    ]
    filtered_files = [majority_vote_files[i] for i in filtered_idxs]
    filtered_labels1 = [exp1_labels[i] for i in filtered_idxs]
    filtered_labels2 = [exp2_labels[i] for i in filtered_idxs]
    filtered_labels3 = [exp3_labels[i] for i in filtered_idxs]

    filtered_idxs_fallback = [
        i for i in range(len(no_maj_vote_files))
        if exp1_other_labels[i] is not None and exp2_other_labels[i] is not None and exp3_other_labels[i] is not None
    ]
    filtered_files_fallback = [no_maj_vote_files[i] for i in filtered_idxs_fallback]
    filtered_labels1_fallback = [exp1_other_labels[i] for i in filtered_idxs_fallback]
    filtered_labels2_fallback = [exp2_other_labels[i] for i in filtered_idxs_fallback]
    filtered_labels3_fallback = [exp3_other_labels[i] for i in filtered_idxs_fallback]

    file_to_label = dict(zip(test_files, test_labels))
    all_agree_labels = [file_to_label[file] for file in all_agree_files]

    all_test_files = majority_vote_files + no_maj_vote_files
    all_test_labels_exp1 = exp1_labels + exp1_other_labels
    all_test_labels_exp2 = exp2_labels + exp2_other_labels
    all_test_labels_exp3 = exp3_labels + exp3_other_labels

    evaluation_sets = {
                        'eval test files':[test_files, test_labels],
                        'eval all experts agree':[all_agree_files, all_agree_labels],
                        'eval majority Vote':[majority_vote_files, gt_vote_labels],
                        #'eval filtered files exp1':[filtered_files, filtered_labels1],
                        #'eval filtered files exp2':[filtered_files, filtered_labels2],
                        #'eval filtered files exp3':[filtered_files, filtered_labels3],
                        #'eval filtered no majority exp1': [filtered_files_fallback, filtered_labels1_fallback],
                        #'eval filtered no majority exp2': [filtered_files_fallback, filtered_labels2_fallback],
                        #'eval filtered no majority exp3': [filtered_files_fallback, filtered_labels3_fallback]
                       }

    if pretraining:
        if inject_noise:
            new_train_labels, noise_log = inject_label_noise(
                train_files=train_files,
                train_labels=train_labels,
                class_lookup=class_lookup,
                groupings_dict=groupings_dict,
                noise_percentages=noise_percentages
            )
            train_labels = new_train_labels
            df = pd.DataFrame(noise_log)
            df.to_csv(os.path.join(iteration_folder, "injected_label_noise_log.csv"), index=False)
            print(f'injected noise: {df.shape[0]}')

        #Pretraining_files, pretraining_labels = [], []
        #Pretraining_files.extend(train_files)
        #Pretraining_labels.extend(train_labels)
        #Pretraining_files.extend(val_files)
        #Pretraining_labels.extend(val_labels)
        #Utils.write_dict_to_json({'train': pretraining_files, 'labels': pretraining_labels},
        #Ssl_folder, 'pretraining_files_labels.json')
        history, ellapsed_time = ssl_training(train_files, train_labels, val_files, val_labels, ssl_folder)
        utils.write_dict_to_json({'pretraining_time':ellapsed_time}, ssl_folder, 'pretraining_info.json')

    original_train_labels = train_labels.copy()

    if os.path.exists(os.path.join(iteration_folder, "injected_label_noise_log.csv")):
        df = pd.read_csv(os.path.join(iteration_folder, "injected_label_noise_log.csv"))
        data_dict = df.set_index('file').to_dict(orient='index')
        cnt_noise_in_train = 0
        for i in range(len(train_files)):
            if train_files[i] in data_dict:
                val = data_dict[train_files[i]]
                noise_label = reversed_class_lookup[val['new_class']]
                curr_label = train_labels[i]
                train_labels[i] = noise_label
                cnt_noise_in_train += 1
        print(f'Noise injected into train: {cnt_noise_in_train}')

    if correction:
        run_self_correction_pipeline(train_files, train_labels, val_files, val_labels,
                                     reversed_class_lookup, class_lookup,
                                     save_folder, ssl_folder,
                                     correction_folder, dual_dict,
                                     os.path.join(iteration_folder, "injected_label_noise_log.csv"))

    #plot_correction_statistics(correction_folder)

    if use_corrections or remove_as_noisy_labels:
        #After assessing corrections, set correction folder to use for the following analyses:
        epoch_correction_folder = os.path.join(iteration_folder, f'correction_{use_epoch_weights}')
        corrected_train_labels, files_needing_correction_train = get_corrected_labels(epoch_correction_folder, train_files,
                                                                                      train_labels, class_lookup)
        corrected_val_labels, files_needing_correction_val = get_corrected_labels(epoch_correction_folder, val_files,
                                                                                  val_labels, class_lookup)
        corrected_test_labels, files_needing_correction_test = get_corrected_labels(epoch_correction_folder, test_files,
                                                                                    test_labels, class_lookup)

        print(f'Possible corrections (train): {len(files_needing_correction_train)}')
        print(f'Possible corrections (val): {len(files_needing_correction_val)}')
        print(f'Possible corrections (test): {len(files_needing_correction_test)}')

        count_equal = sum(1 for a, b in zip(original_train_labels, train_labels) if a == b)
        count_not_equal = sum(1 for a, b in zip(original_train_labels, train_labels) if a != b)

        count_equal1 = sum(1 for a, b in zip(corrected_train_labels, train_labels) if a == b)
        count_not_equal1 = sum(1 for a, b in zip(corrected_train_labels, train_labels) if a != b)

        count_equal2 = sum(1 for a, b in zip(original_train_labels, corrected_train_labels) if a == b)
        count_not_equal2 = sum(1 for a, b in zip(original_train_labels, corrected_train_labels) if a != b)

    if use_corrections:
        #Replace gt labels with corrected labels:
        train_labels = corrected_train_labels
        #val_labels = corrected_val_labels
        #test_labels = corrected_test_labels

        utils.write_dict_to_json(files_needing_correction_train,
                                 epoch_correction_folder, 'files_needing_correction_train.json')
        utils.write_dict_to_json(files_needing_correction_val,
                                 epoch_correction_folder, 'files_needing_correction_val.json')
        utils.write_dict_to_json(files_needing_correction_test,
                                 epoch_correction_folder, 'files_needing_correction_test.json')
        print(f'Corrected counts (train): {len(files_needing_correction_train)}')
        #print(f'Corrected counts (val): {len(files_needing_correction_val)}')
        #print(f'Corrected counts (test): {len(files_needing_correction_test)}')

    elif remove_as_noisy_labels:
        filtered_train_files = []
        filtered_train_labels = []
        #filtered_val_files = []
        #filtered_val_labels = []

        for file, label in zip(train_files, train_labels):
            if file not in files_needing_correction_train:
                filtered_train_files.append(file)
                filtered_train_labels.append(label)

        #for file, label in zip(val_files, val_labels):
        #if file not in files_needing_correction_val:
        #Filtered_val_files.append(file)
        #Filtered_val_labels.append(label)

        train_files = filtered_train_files
        train_labels = filtered_train_labels
        #val_files = filtered_val_files
        #val_labels = filtered_val_labels

    #ABLATION - commented out
    #if use_corrections == True and remove_as_noisy_labels == False:
    #df = pd.read_csv(os.path.join(iteration_folder, "injected_label_noise_log.csv"))
    #data_dict = df.set_index('file').to_dict(orient='index')
    #cnt_replaced = 0
    #for i in range(len(train_files)):
    #if train_files[i] in data_dict and train_files[i] in files_needing_correction_train:
    #val = data_dict[train_files[i]]
    #orig_label = reversed_class_lookup[val['original_class']]
    #Train_labels[i] = orig_label
    #Cnt_replaced += 1
    #print(f'REPLACED: {cnt_replaced}')

    #train_labels = inject_from_noise_file(iteration_folder, train_files, train_labels, reversed_class_lookup)

    #Downstream:
    #Using imageNet weights but trained on corrected labels
    if downstream_with_imagenet or downstream_with_random_init:
        random.seed(444)

        (train_ds, val_ds,
         test_ds, test_images_ds) = get_tf_datasets(train_files, train_labels,
                                                    val_files, val_labels,
                                                    test_files, test_labels, augmentation=False)
        #Training info
        training_info_dict = {}
        run_losses, run_epochs = [], []
        run_val_losses, run_val_epochs = [], []

        downstream_folder = downstream_with_imagenet_folder
        if downstream_with_random_init:
            downstream_folder = downstream_with_random_init_folder

        model = None
        if downstream_training:
            #Train from scratch or fine-tune happens here

            model, history = train_model(train_ds=train_ds, val_ds=val_ds, output_folder=downstream_folder,
                                             percentage=train_percentage, run_index=0, use_image_net_weights=downstream_with_imagenet)

            loss = np.array(history.history['loss']).astype(float)
            val_loss = np.array(history.history['val_loss']).astype(float)
            run_losses.append(np.min(loss))
            run_epochs.append(int(np.argmin(loss)) + 1)  #0-based index but epoch is 1-based
            run_val_losses.append(np.min(val_loss))
            run_val_epochs.append(int(np.argmin(val_loss)) + 1)

            training_info_dict['run_losses'] = run_losses
            training_info_dict['run_epochs'] = run_epochs
            training_info_dict['run_val_losses'] = run_val_losses
            training_info_dict['run_val_epochs'] = run_val_epochs
            training_info_dict['num_train'] = len(train_files)
            training_info_dict['num_val'] = len(val_files)
            training_info_dict['num_test'] = len(test_files)
            utils.write_dict_to_json(training_info_dict, downstream_folder, f'training_info.json')

        tf.keras.backend.clear_session()

        #Get existing weights from previous training and make predictions on test dataset
        model_file = os.path.join(downstream_folder, f'model_run{0}_perc{train_percentage}.h5')
        for key, val in evaluation_sets.items():
            eval_folder = os.path.join(downstream_folder, key)
            evaluate(eval_folder, class_lookup, model_file, val[0], val[1], 0, train_percentage, key)
        evaluate_multi(os.path.join(downstream_folder, 'eval match any expert'), class_lookup, model_file, all_test_files,
                       all_test_labels_exp1, all_test_labels_exp2, all_test_labels_exp3, 0, train_percentage, 'eval match any expert')

    #Using pretrained weights from supervised contrastive
    if downstream_with_pretrainedweights or downstream_with_linear:
        #pretrained_model_file = os.path.join(ssl_folder, 'model_ep_150.h5')
        pretrained_model_file = os.path.join(ssl_folder, f'model_ep_{use_epoch_weights}.h5')
        if use_epoch_weights == 0:
            pretrained_model_file = os.path.join(ssl_folder, f'best_model.h5')

        random.seed(444)

        (train_ds, val_ds,
         test_ds, test_images_ds) = get_tf_datasets(train_files, train_labels,
                                                    val_files, val_labels,
                                                    test_files, test_labels, augmentation = False)
        #Training info
        training_info_dict = {}
        run_losses, run_epochs = [], []
        run_val_losses, run_val_epochs = [], []

        downstream_folder = downstream_with_pretrained_folder
        if downstream_with_linear:
            downstream_folder = downstream_with_linear_folder

        model = None
        if downstream_training:

            #Train from scratch or fine-tune happens here
            model, history = fine_tune_model(train_ds=train_ds, val_ds=val_ds, output_folder=downstream_folder,
                                             percentage=train_percentage, run_index=0, pretrained_model_file = pretrained_model_file,
                                             num_train = len(train_files), linear_eval=downstream_with_linear)

            loss = np.array(history.history['loss']).astype(float)
            val_loss = np.array(history.history['val_loss']).astype(float)
            run_losses.append(np.min(loss))
            run_epochs.append(int(np.argmin(loss)) + 1)  #0-based index but epoch is 1-based
            run_val_losses.append(np.min(val_loss))
            run_val_epochs.append(int(np.argmin(val_loss)) + 1)

            training_info_dict['run_losses'] = run_losses
            training_info_dict['run_epochs'] = run_epochs
            training_info_dict['run_val_losses'] = run_val_losses
            training_info_dict['run_val_epochs'] = run_val_epochs
            training_info_dict['num_train'] = len(train_files)
            training_info_dict['num_val'] = len(val_files)
            training_info_dict['num_test'] = len(test_files)
            utils.write_dict_to_json(training_info_dict, downstream_folder, f'training_info.json')

        tf.keras.backend.clear_session()

        #Get existing weights from previous training and make predictions on test dataset
        model_file = os.path.join(downstream_folder, f'model_run{0}_perc{train_percentage}.h5')
        for key, val in evaluation_sets.items():
            eval_folder = os.path.join(downstream_folder, key)
            evaluate(eval_folder, class_lookup, model_file, val[0], val[1], 0, train_percentage, key)
        evaluate_multi(os.path.join(downstream_folder, 'eval match any expert'), class_lookup, model_file, all_test_files,
                 all_test_labels_exp1, all_test_labels_exp2, all_test_labels_exp3, 0, train_percentage, 'eval match any expert')

    

    exp_df = create_plots(
        base_dir="/mnt/ssd/preshen/backup",
        figures_dir="/mnt/ssd/preshen/backup/run_view_with_noise_FIGURES",
        folder_list_json=None,
    )

    run_best_config_significance_tests(
        base_dir="/mnt/ssd/preshen/backup",
        folder_list_json="/mnt/ssd/preshen/backup/folders_exp.json",
        out_dir=os.path.join("/mnt/ssd/preshen/backup", "run_view_with_noise_STATS"),
        architectures=("xception", "convnexttiny", "swintransformerv2tiny", "efficientnetv2s"),
        methods=("supcon", "logsum", "dropcon_k1_v2", "dropcon_k1_v3"),
        method_pairs=(
            ("supcon", "logsum"),
            ("supcon", "dropcon_k1_v2"),
            ("logsum", "dropcon_k1_v2"),
            #Add these if you also want v3 comparisons:
            ("supcon", "dropcon_k1_v3"),
            ("logsum", "dropcon_k1_v3"),
            ("dropcon_k1_v2", "dropcon_k1_v3"),
        ),
        noise_levels=(0, 10, 20, 30, 40, 50),
        subfolder="downstream_with_pretrained_model_0",
        eval_folders=("eval test files",),  #Add others if you want
        pred_file="all_predictions_0.json",
        require_all_runs=True,
        n_perm=5000,
        n_boot=2000,
        seed=0,
    )    

    #evaluate_hierarchical_accuracy(f'run_view_with_noise_Aug-True_T03_logsum/ap0_pl0_ps0_do0_mm0/downstream_with_pretrained_model_0/eval test files/all_predictions_0.json', class_lookup)
    #plot_mismatch_analysis_from_json(f'run_view_with_noise_Aug-True_T03_logsum/ap0_pl0_ps0_do0_mm0/correction_0/kmeans_mismatches_train.json', 'run_view_with_noise_FIGURES')
    #generate_confusion_matrices_from_prediction_json(
    #prediction_json_path='run_view_with_noise_Aug-True_T03_logsum/ap0_pl0_ps0_do0_mm0/downstream_with_pretrained_model_0/eval test files/all_predictions_0.json',
    #expert1_labels=all_test_labels_exp1,
    #expert2_labels=all_test_labels_exp2,
    #expert3_labels=all_test_labels_exp3,
    #test_files = all_test_files,
    #class_lookup=class_lookup,
    #output_folder='run_view_with_noise_FIGURES/conf_matrices'
    #)
    #generate_confusion_matrices_intersection_subset(
    #prediction_json_path='run_view_with_noise_Aug-True_T03_logsum/ap0_pl0_ps0_do0_mm0/downstream_with_pretrained_model_0/eval test files/all_predictions_0.json',
    #expert1_labels=all_test_labels_exp1,
    #expert2_labels=all_test_labels_exp2,
    #expert3_labels=all_test_labels_exp3,
    #test_files = all_test_files,
    #class_lookup=class_lookup,
    #output_folder='run_view_with_noise_FIGURES/conf_matrices_intersection_4882'
    #)
    #save_disagreement_examples(
    #prediction_json_path='run_view_with_noise_Aug-True_T03_logsum/ap0_pl0_ps0_do0_mm0/downstream_with_pretrained_model_0/eval test files/all_predictions_0.json',
    #expert1_labels=all_test_labels_exp1,
    #expert2_labels=all_test_labels_exp2,
    #expert3_labels=all_test_labels_exp3,
    #test_files=all_test_files,
    #class_lookup=class_lookup,
    #output_folder='run_view_with_noise_FIGURES/conf_matrices_intersection_4882',
    #num_samples=20,
    #seed=123
    #)
    #compute_cohens_kappa_across_noise_intersection_experts(
    #base_prediction_json_path='run_view_with_noise_Aug-True_T03_logsum/ap0_pl0_ps0_do0_mm0/downstream_with_pretrained_model_0/eval test files/all_predictions_0.json',
    #expert1_labels=all_test_labels_exp1,
    #expert2_labels=all_test_labels_exp2,
    #expert3_labels=all_test_labels_exp3,
    #test_files=all_test_files,
    #class_lookup=class_lookup,
    #output_dir='run_view_with_noise_FIGURES/CohenKappa' #optional; defaults next to the predictions JSON
    #)
    #compute_cohens_kappa_across_noise_intersection_experts(
    #base_prediction_json_path='run_view_with_noise_Aug-True_T005_supcon/ap0_pl0_ps0_do0_mm0/downstream_with_rand_init/eval test files/all_predictions_0.json',
    #expert1_labels=all_test_labels_exp1,
    #expert2_labels=all_test_labels_exp2,
    #expert3_labels=all_test_labels_exp3,
    #test_files=all_test_files,
    #class_lookup=class_lookup,
    #output_dir='run_view_with_noise_FIGURES/CohenKappa_rand_init' #optional; defaults next to the predictions JSON
    #)
    #compute_mwi_across_noise_intersection_experts(
    #base_prediction_json_path='run_view_with_noise_Aug-True_T03_logsum/ap0_pl0_ps0_do0_mm0/downstream_with_pretrained_model_0/eval test files/all_predictions_0.json',
    #expert1_labels=all_test_labels_exp1,
    #expert2_labels=all_test_labels_exp2,
    #expert3_labels=all_test_labels_exp3,
    #test_files=all_test_files,
    #class_lookup=class_lookup,
    #output_dir='run_view_with_noise_FIGURES/mWI' #optional; defaults next to the predictions JSON
    #)
    #compute_mwi_across_noise_intersection_experts(
    #base_prediction_json_path='run_view_with_noise_Aug-True_T005_supcon/ap0_pl0_ps0_do0_mm0/downstream_with_rand_init/eval test files/all_predictions_0.json',
    #expert1_labels=all_test_labels_exp1,
    #expert2_labels=all_test_labels_exp2,
    #expert3_labels=all_test_labels_exp3,
    #test_files=all_test_files,
    #class_lookup=class_lookup,
    #output_dir='run_view_with_noise_FIGURES/mWI_rand_init' #optional; defaults next to the predictions JSON
    #)

    #annotate_selection_pvals("run_view_with_noise_FIGURES_temp/selection_pvals_eval_test_files.csv")

    print('Completed.')

if __name__ == "__main__":
    main()

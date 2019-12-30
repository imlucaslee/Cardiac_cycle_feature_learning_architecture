from __future__ import division
import tensorflow as tf
import numpy as np
import os 
import dicom
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from skimage.util import img_as_float
from skimage.segmentation import slic
import os
import sys
import random
import scipy.ndimage
import nibabel as nib

SPINE_LABELS = {
    'none':(0,'background'),
    'dn':(1, 'Normal Disc')

}


RANDOM_SEED = 4242
SAMPLES_PER_FILES = 300

def get_groundtruth_from_xml(xml):
    labels = []
    labels_text = []
    instance = []
    coordinates_class = [] #The key of this dictionary is the class and values are class' coordinates.
    coordinates_instance = {} #The key of this dictionary is the class and values are instance' coordinates.
    tree = ET.parse(xml)
    root = tree.getroot()
    rows = root.find('imagesize').find('nrows').text
    columns = root.find('imagesize').find('ncols').text
    shape = [int(rows),int(columns),int(1)]
    masks= np.array([rows,columns])
    for object in root.findall('object'):
        coordinate = []
        if object.find('deleted').text != 1:
            label = object.find('name').text  # class-wise character groundtruth
            label_int = int(SPINE_LABELS[label][0]) # class-wise number groundtruth
            #append to lists
            labels.append(label_int)
            labels_text.append(label.encode('ascii'))
            
            instance_label_int = int(object.find('id').text) # instance-wise number groundtruth
            instance.append(instance_label_int)            
            polygon = object.find('polygon')
            for pt in polygon.findall('pt'):
                x = int(pt.find('x').text)
                y = int(pt.find('y').text)
                coordinate.append((x,y))
            coordinates_class.append(coordinate)
            coordinates_instance[instance_label_int] = coordinate
    return labels, labels_text, instance, shape, coordinates_class, coordinates_instance
                
def groundtruth_to_mask(xml):
    
    labels, labels_text, instance, shape, coordinates_class, coordinates_instance = get_groundtruth_from_xml(xml)
    #print shape
    #draw image first and then using numpy to matrix.
    img_instance = Image.new('L', (shape[0], shape[1]), 0)
    img_class = Image.new('L', (shape[0], shape[1]), 0)
    for i in coordinates_instance:###instance mask
        polygon_instance = coordinates_instance[i]
        #polygon = [(1,1), (5,1), (5,9),(3,2),(1,1)]
        ImageDraw.Draw(img_instance).polygon(polygon_instance, outline=0, fill=i)    
    for j,k in enumerate(coordinates_class):
        polygon_class = k
        ImageDraw.Draw(img_class).polygon(polygon_class, outline=0, fill=labels[j])
    mask_instance = np.array(img_instance)
    mask_class = np.array(img_class)
    return mask_instance, mask_class, shape, labels, labels_text, instance

def get_image_superpixels_data_from_dicom(dm):
    dm = dicom.read_file(dm)
    x = 512
    y = 512
    xscale = x/dm.Rows
    yscale = y/dm.Columns
    image_data = np.array(dm.pixel_array)
    #image_data = np.float32(image_data)
    image_data = scipy.ndimage.interpolation.zoom(image_data, [xscale,yscale])
    print(image_data.shape)
    #image = img_as_float(image_data)
    superpixels = slic(image_data, n_segments = 2000, compactness=0.01, max_iter=10)
    return image_data, superpixels
def get_image_data_from_dicom(dm):
    dm = dicom.read_file(dm)
    x = 512
    y = 512
    xscale = x/dm.Rows
    yscale = y/dm.Columns
    image_data = np.array(dm.pixel_array)
    #image_data = np.float32(image_data)
    image_data = scipy.ndimage.interpolation.zoom(image_data, [xscale,yscale])
    print(image_data.shape)
    #image = img_as_float(image_data)
    #superpixels = slic(image_data, n_segments = 2000, compactness=0.01, max_iter=10)
    return image_data

def get_image_data_from_nii(dataset_dir,i,j):
    img_build = dataset_dir + i + "/" + j + ".nii"
    lab_build = dataset_dir + i + "/" + "5" + ".nii"
    image_data = np.array(nib.load(img_build).get_data())[2:35,:,:]
    label_data = np.array(nib.load(lab_build).get_data())[2:35,:,:]
    
    #image_data = np.float32(image_data)
    image_data = scipy.ndimage.interpolation.zoom(image_data, [1,1,1])
    label_data = scipy.ndimage.interpolation.zoom(label_data, [1,1,1])
    label_data.dtype = "uint8"
    print(image_data.dtype)
    print(image_data.shape)
    print(label_data.dtype)
    print(label_data.shape)
    #image = img_as_float(image_data)
    #superpixels = slic(image_data, n_segments = 2000, compactness=0.01, max_iter=10)
    return image_data, label_data

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _convert_to_example(image_data, mask_class):

    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image.
      labels: list of integers, identifier for the ground truth;
      instance: instance labels.
      labels_text: list of strings, human-readable labels.
      mask_instance: numpy matrix of instance.
      mask_class: numpy matrix of class.   
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/image_data':_bytes_feature(image_data.tostring()),
        'image/mask_class':_bytes_feature(mask_class.tostring())
    }))
    return example

def _add_to_tfrecord(dataset_dir, i, j, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.
    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """

    image_data, mask_class= get_image_data_from_nii(dataset_dir,i,j)
    example = _convert_to_example(image_data, mask_class)
    tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(output_dir, name):
    return '%s/%s.tfrecord' % (output_dir, name)

def run(dataset_dir, output_dir,  shuffling=False):
    """
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    #path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(dataset_dir))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)
    name= "train"

    tf_filename  = _get_output_filename(output_dir, name)   
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:

        for i in range(7):
            for j in range(4):
                _add_to_tfrecord(dataset_dir, str(i+1), str(j+1),tfrecord_writer)
    print('\nFinished converting the spine segmentation dataset!')


run("C:/Users/lucaslee/Desktop/1/Multi-modalityMRISpineImage/","C:/Users/lucaslee/Desktop/1/Multi-modalityMRISpineImage/output")
        
        

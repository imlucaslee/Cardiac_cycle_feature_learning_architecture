from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from nets import PathNet
import tf_utils
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Fold is one of five folds cross-validation. 
#Fold = 2
# =========================================================================== #
# Evaluation Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'Fold', '1', 'Flod of 5-fold cross validation.')


tf.app.flags.DEFINE_string(
    'eval_dir', 'tmp/tfmodels_gan_lstm_%s/'%FLAGS.Fold, 'Directory where the results are saved to.')

tf.app.flags.DEFINE_string('checkpoint_dir', 'tmp/tfmodels_gan_lstm_%s/'%FLAGS.Fold,
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")


tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #

tf.app.flags.DEFINE_string(
    'dataset_name', 'spine_segmentation', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test_%s_fold'%FLAGS.Fold, 'The name of the train/test split and five folds.')
tf.app.flags.DEFINE_string(
    'dataset_dir', './datasets/tfrecords_spine_segmentation/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'batch_size', 29, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'num_samples', 58, 'The number of samples in the testing set.')
tf.app.flags.DEFINE_integer(
    'num_readers', 10,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')



def main(_):
    
    if not FLAGS.dataset_dir:
        
            raise ValueError('You must supply the dataset directory with --dataset_dir')
            
    with tf.Graph().as_default():
        
        # Select the dataset.
        dataset = FLAGS.dataset_dir + '%s_%s.tfrecord' %(FLAGS.dataset_name,FLAGS.dataset_split_name) 
        
        with tf.name_scope('input'):
            
            filename_queue = tf.train.string_input_producer([dataset], num_epochs= 1)
            
            image, mask_class = tf_utils.read_and_decode_for_lstm(filename_queue, batch_size = FLAGS.batch_size,capacity=20 * FLAGS.batch_size,num_threads=FLAGS.num_readers,min_after_dequeue=10 * FLAGS.batch_size, is_training=False)
            
        fc_2 = PathNet.motion_feature_fusion_extraction_regression(image, batch_size=FLAGS.batch_size, reuse=False,is_training=False)  


        pred = fc_2

        gt = mask_class
        
                   
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        
        sess = tf.Session()

        sess.run(init_op)
        
        

        loader = tf.train.Saver()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            loader.restore(sess, ckpt.model_checkpoint_path)
            print("Restored model parameters from {}".format(ckpt))
        else: 
            print('No checkpoint file found')
            
        num_iter = int(math.ceil(FLAGS.num_samples / FLAGS.batch_size))

        # Iterate over training steps.
        for step in range(num_iter):
    
            images,preds,gts = sess.run([image,pred,gt])
            print("shape:" + str(gts.shape) + str(preds.shape) + str(images.shape))
            a_max = 0.0
            for i in range(29):
                a = gts[i] - preds[i]
                print("gts-preds = " + str(a) + str(step))
                if a < 0:
                    a = -a
                a_max = a_max + a
            print("a_max:" + str(a_max))
            a_avg = a_max / 29

            std = 0.0
            print("a_avg:" + str(a_avg))
            for j in range(29):
                a = gts[j] - preds[j]
                b = (a_avg - a) * (a_avg - a)
                std = std + b
            std = std / (29 - 1)
            std = math.sqrt(std)
            print("std:" + str(std))
            
            gts_max = 0.0
            gts_avg = 0.0
            preds_max = 0.0
            preds_avg = 0.0
            for m in range(29):
                gts_max = gts_max + gts[m]
                preds_max = preds_max + preds[m]
            gts_avg = gts_max / 29
            preds_avg = preds_max / 29

            cov_max = 0.0
            cov_avg = 0.0
            var_gts_max = 0.0
            var_gts_avg = 0.0
            var_preds_max = 0.0
            var_preds_avg = 0.0
            var = 0.0
            cc = 0.0
            for n in range(29):
                cov_max = cov_max + ((gts_avg - gts[n]) * (preds_avg - preds[n]))
                var_gts_max = var_gts_max + (gts_avg - gts[n]) * (gts_avg - gts[n])
                var_preds_max = var_preds_max + (preds_avg - preds[n]) * (preds_avg - preds[n])
            cov_avg = cov_max / 29
            var_gts_avg = var_gts_max / 29
            var_preds_avg = var_preds_max / 29
            var = var_gts_avg * var_preds_avg
            var = math.sqrt(var)
            cc = cov_avg / var
            print("cc:" + str(cc))
        
        coord.request_stop()
        coord.join(threads)
        sess.close()                          
       
    
    
if __name__ == '__main__':

    tf.app.run()        
        
       
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


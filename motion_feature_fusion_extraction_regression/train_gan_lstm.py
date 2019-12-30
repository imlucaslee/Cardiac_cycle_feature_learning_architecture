
import os
import tensorflow as tf
#from tensorflow.python.ops import control_flow_ops
from nets import PathNet,losses
import tf_utils

# Fold is one of five folds cross-validation. 
Fold = 1
# =========================================================================== #
# Model saving Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', 'tmp/tfmodels_gan_lstm_%s/'%Fold,
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1, 'GPU memory fraction to use.')

tf.app.flags.DEFINE_float(
    'weight', 1, 'The weight is between cross entropy loss and metric loss.')
# =========================================================================== #
# Dataset Flags.
# =========================================================================== #

tf.app.flags.DEFINE_string(
    'dataset_name', 'spine_segmentation', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train_%s_fold'%Fold, 'The name of the train/test split.')#spine_segmentation_train_7_class.tfrecord
tf.app.flags.DEFINE_string(
    'dataset_dir', './datasets/tfrecords_spine_segmentation/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'batch_size', 29, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'num_samples', 87, 'The number of samples in the training set.')
tf.app.flags.DEFINE_integer(
    'num_readers', 10,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.94,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.96, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float( 
    'num_epochs_per_decay', 10.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('num_epochs', 500,
                            'The number of training epochs.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', 'tmp/tfmodels_gan_lstm_%s/'%Fold,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', None,#['PathNet/feature_embedding','PathNet/logits/'],
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS
# =========================================================================== #
# Main training routine.
# =========================================================================== #


def main(_):
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    ##logging tools
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    with tf.Graph().as_default():
        
        
        # Create global_step.
        with tf.device('/cpu:0'):
            global_step = tf.train.create_global_step()
        
        
        

        # Select the dataset.
        dataset = FLAGS.dataset_dir + '%s_%s.tfrecord' %(FLAGS.dataset_name,FLAGS.dataset_split_name) 
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):            
                filename_queue = tf.train.string_input_producer([dataset], num_epochs=FLAGS.num_epochs)
                image,mask_class = tf_utils.read_and_decode_for_lstm(filename_queue, batch_size = FLAGS.batch_size,\
                                                                       capacity=20 * FLAGS.batch_size,\
                                                                       num_threads=FLAGS.num_readers,\
                                                                       min_after_dequeue=10 * FLAGS.batch_size, is_training=True)
                


        D_logit_real = PathNet.motion_feature_fusion_extraction_regression(image, batch_size=FLAGS.batch_size)


        with tf.name_scope('d_loss'):
            

            D_loss_real = tf.reduce_mean(tf.abs(mask_class-D_logit_real))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        
       # print (g_vars)
        
        learning_rate = tf_utils.configure_learning_rate(FLAGS,FLAGS.num_samples, global_step)
        
        
        optimizer = tf.train.AdamOptimizer(beta1=0.5, learning_rate=0.001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        
        #Note: when training, the moving_mean and moving_variance need to be updated.
        with tf.control_dependencies(update_ops):
          
            train_op_D = optimizer.minimize(D_loss_real, global_step=global_step, var_list=d_vars)
        # The op for initializing the variables.                                                            
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        
        
        # Create a session for running operations in the Graph.

        sess = tf.Session()
        
        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)
        
        
        #Include max_to_keep=5
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            exclude_list = FLAGS.ignore_missing_vars
            variables_list  = tf.contrib.framework.get_variables_to_restore(exclude=exclude_list)
            restore = tf.train.Saver(variables_list)
            restore.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            global_step = int(global_step)

        else: global_step = 0    
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        
        # Save models.
        if not tf.gfile.Exists(FLAGS.train_dir):
                #tf.gfile.DeleteRecursively(FLAGS.train_dir)
                tf.gfile.MakeDirs(FLAGS.train_dir)
                
                
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        
        
        try:
            step = global_step # TODO: Continue to train the model, if the checkpoints are exist.
            while not coord.should_stop():
                #start_time = time.time()
                   
                _,dlr,lr = sess.run([train_op_D, D_loss_real,learning_rate])    

                                                 
                #duration = time.time() - start_time
                
                if step % 1 == 0:
                    print("Discrimator loss of real = %.5f, step = %d, learning rate = %.6f"%(dlr, step, lr))
                    print(str(dlr.dtype))

                step += 1
                
                if step % 1 == 0: #
                    

                    
                    saver.save(sess, checkpoint_path, global_step=step)
                                                                      
        except tf.errors.OutOfRangeError:
                
            print('Done training for %d epochs, %d steps.'%(FLAGS.num_epochs, step))  
            saver.save(sess, checkpoint_path, global_step=step) 

            print('Model is saved as %s'%checkpoint_path)
            
        finally:
            

            coord.request_stop()


        coord.join(threads)
        sess.close()                                                      
                                                                         

if __name__ == '__main__':
    tf.app.run()    

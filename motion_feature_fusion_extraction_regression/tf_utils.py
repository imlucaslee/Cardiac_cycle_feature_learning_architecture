from __future__ import print_function
import os
from pprint import pprint

import tensorflow as tf
#from tensorflow.contrib.slim.python.slim.data import parallel_reader

def read_and_decode_for_lstm(filename_queue, batch_size = 8, capacity=30, num_threads=1, min_after_dequeue=10, is_training=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'image/image_data': tf.FixedLenFeature([], tf.string),
        'image/mask_class': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(features['image/image_data'], tf.float64)
    mask_class = tf.decode_raw(features['image/mask_class'], tf.float64)
    image = tf.reshape(image, [2,80, 80, 1])
    image = tf.cast(image,tf.float32)  
    mask_class=tf.reshape(mask_class, [1])
    mask_class = tf.cast(mask_class,tf.float32)
     
    #resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                          # target_height=300,
                                          # target_width=300)
    #image_size_const = tf.constant((512, 512, 1), dtype=tf.int32)
    if is_training is True:
        
        image,mask_class = tf.train.shuffle_batch( [image,mask_class], batch_size=batch_size,
                                                 capacity=capacity,
                                                 num_threads=num_threads,
                                                 min_after_dequeue=min_after_dequeue)
    else:
        image,mask_class = tf.train.batch([image,mask_class], batch_size=batch_size,\
                                                        num_threads=num_threads,capacity=capacity)
        print("Testttttttttttttttttttttttttttttttttt: "+str(image.shape) + str(mask_class.shape))
    return image, mask_class


def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.
    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r

# =========================================================================== #
# Training utils.
# =========================================================================== #
def print_configuration(flags, ssd_params, data_sources, save_dir=None):
    """Print the training configuration.
    """
    def print_config(stream=None):
        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation flags:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(flags, stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# SSD net parameters:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(dict(ssd_params._asdict()), stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        data_files = parallel_reader.get_data_files(data_sources)
        pprint(sorted(data_files), stream=stream)
        print('', file=stream)

    print_config(None)
    # Save to a text file as well.
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, 'training_config.txt')
        with open(path, "w") as out:
            print_config(out)
            
def configure_learning_rate(flags, num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch / flags.batch_size *
                      flags.num_epochs_per_decay)

    if flags.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(flags.learning_rate,
                                          global_step,
                                          decay_steps,
                                          flags.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif flags.learning_rate_decay_type == 'fixed':
        return tf.constant(flags.learning_rate, name='fixed_learning_rate')
    elif flags.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(flags.learning_rate,
                                         global_step,
                                         decay_steps,
                                         flags.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         flags.learning_rate_decay_type)
def configure_optimizer(flags, learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    """
    if flags.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=flags.adadelta_rho,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=flags.adagrad_initial_accumulator_value)
    elif flags.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=flags.adam_beta1,
            beta2=flags.adam_beta2,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=flags.ftrl_learning_rate_power,
            initial_accumulator_value=flags.ftrl_initial_accumulator_value,
            l1_regularization_strength=flags.ftrl_l1,
            l2_regularization_strength=flags.ftrl_l2)
    elif flags.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=flags.momentum,
            name='Momentum')
    elif flags.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=flags.rmsprop_decay,
            momentum=flags.rmsprop_momentum,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', flags.optimizer)
    return optimizer


def get_init_fn(flags):
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.
    Returns:
      An init function run by the supervisor.
    """
    if flags.checkpoint_path is None:
        return None
    # Warn the user if a checkpoint exists in the train_dir. Then ignore.
    if tf.train.latest_checkpoint(flags.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % flags.train_dir)
        return None

    exclusions = []
    if flags.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in flags.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if flags.checkpoint_model_scope is not None:
        variables_to_restore = \
            {var.op.name.replace(flags.model_name,
                                 flags.checkpoint_model_scope): var
             for var in variables_to_restore}


    if tf.gfile.IsDirectory(flags.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(flags.checkpoint_path)
    else:
        checkpoint_path = flags.checkpoint_path
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, flags.ignore_missing_vars))

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=flags.ignore_missing_vars)


def get_variables_to_train(flags):
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if flags.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

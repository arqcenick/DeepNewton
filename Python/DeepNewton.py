import tensorflow as tf
import numpy as np


def DN_model(input_tensor):
	with tf.device("/gpu:0"):
		weights = []
		tensor = None

        conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/6)))
		conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
		weights.append(conv_00_w)
		weights.append(conv_00_b)
		tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))

        conv_00_w = tf.get_variable("conv_01_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/6)))
		conv_00_b = tf.get_variable("conv_01_b", [64], initializer=tf.constant_initializer(0))
		weights.append(conv_00_w)
		weights.append(conv_00_b)
		tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))


        


        conv_00_w = tf.get_variable("conv_00_w", [3,3,1,128], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/6)))
		conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
		weights.append(conv_00_w)
		weights.append(conv_00_b)
		tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))

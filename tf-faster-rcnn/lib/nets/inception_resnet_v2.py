from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg

class inception_resnet_v2(Network):
	"""docstring for inception_resnet_v2"""
	def __init__(self):
		Network.__init__(self)
		self._feat_stride = [16, ]
		self._feat_compress = [1. / float(self._feat_stride[0]), ]
		self._scope = 'InceptionResnetV2'

	def _image_to_head(self, is_training, reuse=None):
		with tf.variable_scope(self._scope, self._scope, [self._image], reuse=reuse):
			with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
				# 149 x 149 x 32
				net = slim.conv2d(self._image, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')

				# 147 x 147 x 32
				net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')

				# 147 x 147 x 64
				net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')

				# 73 x 73 x 64
				net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')

				# 73 x 73 x 80
				net = slim.conv2d(net, 80, 1, padding='VALID', scope='Conv2d_3b_1x1')

				# 71 x 71 x 192
				net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')

				# 35 x 35 x 192
				net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')

				# 35 x 35 x 320
				with tf.variable_scope('Mixed_5b'):
					with tf.variable_scope('Branch_0'):
						tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
					with tf.variable_scope('Branch_1'):
						tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
						tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5, scope='Conv2d_0b_5x5')
					with tf.variable_scope('Branch_2'):
						tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
						tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3, scope='Conv2d_0b_3x3')
						tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3, scope='Conv2d_0c_3x3')
					with tf.variable_scope('Branch_3'):
						tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME', scope='AvgPool_0a_3x3')
						tower_pool_1 = slim.conv2d(tower_pool, 64, 1, scope='Conv2d_0b_1x1')
					net = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)

				# repeat block 35
				net = slim.repeat(net, 10, self._block35, scale=0.17, activation_fn=tf.nn.relu)

				# 33 x 33 x 1088
				with tf.variable_scope('Mixed_6a'):
					with tf.variable_scope('Branch_0'):
						tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
					with tf.variable_scope('Branch_1'):
						tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
						tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3, scope='Conv2d_0b_3x3')
						tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
					with tf.variable_scope('Branch_2'):
						tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
					net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

				# repeat block 17
				with slim.arg_scope([slim.conv2d], rate=1):
					net = slim.repeat(net, 20, self._block17, scale=0.10, activation_fn=tf.nn.relu)

				# 8 x 8 x 2080
				with tf.variable_scope('Mixed_7a'):
					with tf.variable_scope('Branch_0'):
						tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
						tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
					with tf.variable_scope('Branch_1'):
						tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
						tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
					with tf.variable_scope('Branch_2'):
						tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
						tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3, scope='Conv2d_0b_3x3')
						tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
					with tf.variable_scope('Branch_3'):
						tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
					net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)

				# repeat block 8
				net = slim.repeat(net, 9, self._block8, scale=0.20, activation_fn=tf.nn.relu)
				net = self._block8(net, activation_fn=None)

			self._act_summaries.append(net)
			self._layers['head'] = net

			return net

	def _head_to_tail(self, pool5, is_training, reuse=None):
		with tf.variable_scope(self._scope, self._scope, reuse=reuse):
			with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
				with tf.variable_scope('Logits'):
					kernel_size = pool5.get_shape()[1:3]
					if kernel_size.is_fully_defined():
						net = slim.avg_pool2d(pool5, kernel_size, padding='VALID', scope='AvgPool_1a_8x8')
					else:
						net = tf.reduce_mean(pool5, [1, 2], keep_dims=True, name='global_pool')
					net = slim.flatten(net)
					if is_training:
						net = slim.dropout(net, 0.8, is_training=True, scope='dropout')
					logits = slim.fully_connected(net, 1001, activation_fn=None, scope='logits')
					return logits

	def get_variables_to_restore(self, variables, var_keep_dic):
		variables_to_restore = []
		for v in variables:
			# exclude the first conv layer to swap RGB to BGR
			# if v.name == (self._scope + '/conv1/weights:0'):
			# 	self._variables_to_fix[v.name] = v
			# 	continue
			# exclude the conv weights that are fc weights in inception_resnet_v2
			# if v.name == (self._scope + '/logits/weights:0'):
			# 	self._variables_to_fix[v.name] = v
			# 	continue
			if v.name.split(':')[0] in var_keep_dic:
				print('Variables restored: %s' % v.name)
				variables_to_restore.append(v)

		return variables_to_restore

	def fix_variables(self, sess, pretrained_model):
		print('Fix Inception Resnet V2 layers..')
		# with tf.variable_scope('Fix_Inception_Resnet_V2') as scope:
		# 	with tf.device("/cpu:0"):
		# 		Conv2d_1a_3x3_rgb = tf.get_variable("Conv2d_1a_3x3_rgb", [3, 3, 3, 32], trainable=False)
		# 		logits_conv = tf.get_variable("logits_conv", [], trainable=False)
		# 		# restorer_fc = tf.train.Saver({
		# 		# 	self._scope + "/Conv2d_1a_3x3/weights": Conv2d_1a_3x3_rgb,
		# 		# 	self._scope + "/logits/weights": logits_conv})
		# 		restorer_fc = tf.train.Saver({
		# 			self._scope + "/Conv2d_1a_3x3/weights": Conv2d_1a_3x3_rgb})
		# 		restorer_fc.restore(sess, pretrained_model)

		# 		sess.run(tf.assign(
		# 			self._variables_to_fix[self._scope + '/Conv2d_1a_3x3/weights:0'], 
		# 			tf.reverse(Conv2d_1a_3x3_rgb, [2])))

	def _block35(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
		with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
			with tf.variable_scope('Branch_1'):
				tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
				tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
			with tf.variable_scope('Branch_2'):
				tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
				tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
				tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
			mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
			up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
			scaled_up = up * scale
			if activation_fn == tf.nn.relu6:
				scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)
			net += scaled_up
			if activation_fn:
				net = activation_fn(net)
		return net

	def _block17(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
		with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
			with tf.variable_scope('Branch_1'):
				tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
				tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7], scope='Conv2d_0b_1x7')
				tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1], scope='Conv2d_0c_7x1')
			mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
			up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')

			scaled_up = up * scale
			if activation_fn == tf.nn.relu6:
				scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)
			net += scaled_up
			if activation_fn:
				net = activation_fn(net)
		return net

	def _block8(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
		with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
			with tf.variable_scope('Branch_1'):
				tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
				tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3], scope='Conv2d_0b_1x3')
				tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1], scope='Conv2d_0c_3x1')
			mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
			up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')

			scaled_up = up * scale
			if activation_fn == tf.nn.relu6:
				scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)
			net += scaled_up
			if activation_fn:
				net = activation_fn(net)
		return net
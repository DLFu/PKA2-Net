import numpy as np
import glob
import pandas as pd
import tensorflow as tf


#Template_generator
def Template_generator(H, L, radi, ways):
	stride = H/L
	#radi = H/4
	delete = []
	Temp = []
	for i in range (L+1):
		delete.append(i*stride)

	X = np.linspace(0, H-1, H)
	Y = np.linspace(0, H-1, H)

	for n in range(L+1):
		for m in range(L+1):
			if ways == 'L1':
				Z = np.array([
					[
						(radi - np.abs((i-stride*n)) - np.abs((j-stride*m)))
						for j in Y
					]
					for i in X
				])
				for i in range(H):
					for j in range(H):
						if np.abs(i-stride*n) + np.abs(j-stride*m) >radi:
							Z[i,j] = 0

			else:
				Z = np.array([
					[
						(radi**2 - ((i-stride*n) ** 2 + (j-stride*m) ** 2)) ** 0.5
						for j in Y
					]
					for i in X
				])		
				for i in range(H):
					for j in range(H):
						if (i-stride*n)**2 + (j-stride*m)**2 >radi**2:
							Z[i,j] = 0			
			Temp.append(dataset_normalized(Z))
	return Temp


def res_block(inputx,kernel_sum,phase,NAME):
	inputx_new = tf.layers.conv2d(inputx,kernel_sum,3,strides=1,kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=sca),padding='same')
	#inputx_new = tf.layers.conv2d(inputx,4*kernel_sum,1,strides=1,padding='same')
	inputx_new_add = tf.layers.batch_normalization(inputx_new, training=phase)
	#inputx_new_add = switch_norm(inputx_new, NAME)

	inputx_1 = tf.layers.conv2d(inputx_new_add, 4*kernel_sum,1,strides=1,padding='same')
	conv1_1 = tf.nn.relu(inputx_1)

	conv1_2 = tf.layers.conv2d(conv1_1,kernel_sum,1,strides=1,padding='same')

	conv_out = conv1_2 + inputx
	return conv_out


def res_block_down(inputx,kernel_sum,phase,NAME):
	inputx_new = tf.layers.conv2d(inputx,kernel_sum,3,strides=2,kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=sca),padding='same')
	#inputx_new_add = switch_norm(inputx_new, NAME)
	inputx_new_add = tf.layers.batch_normalization(inputx_new, training=phase)

	inputx_1 = tf.layers.conv2d(inputx_new_add, 4*kernel_sum,1,strides=1,padding='same')
	conv1_1 = tf.nn.relu(inputx_1)

	conv1_2 = tf.layers.conv2d(conv1_1,kernel_sum,1,strides=1,padding='same')

	return conv1_2


def SEBS(Tem, input0, size,batch,channel): 
	XHCH = 3
	input = tf.cast(input0, tf.float16)
	in_input = input 
	cn_layer = tf.zeros([batch,size,size,1],dtype=tf.float16)
	Tem = np.reshape(Tem,[1,size,size,-1,1])
	input = tf.reshape(input,[batch,size,size,1,-1])
	in_layer_mul = Tem*input
	Tem1 = 2*Tem/2
	Tem1[Tem1>0] = 1
	tem1_add = tf.reduce_sum(Tem1, axis=[1,2])
	in_layer_GAP = tf.reduce_mean(in_layer_mul, axis=[1,2])/tem1_add
	GAP_ones = tf.ones_like(in_layer_GAP,dtype=tf.float16)
	GAP_zeros = tf.zeros_like(in_layer_GAP,dtype=tf.float16)
	in_layer_GAP_max= tf.where(in_layer_GAP < tf.expand_dims(tf.reduce_max(in_layer_GAP, axis=-2), -2),GAP_zeros,GAP_ones)
	in_layer_GAP_max = tf.reshape(in_layer_GAP_max,[batch,1,1,-1,1,channel])
	in_layer_all = tf.expand_dims(in_layer_mul, -2)
	# current_featuremap_max = in_layer_GAP_max*in_layer_all
	# current_featuremap_max = tf.reduce_sum(current_featuremap_max, axis = -3)
	# current_featuremap_max = tf.reshape(current_featuremap_max,[batch,size,size,-1])	
	in_layer_GAP_kmax0 = in_layer_GAP
	atten = in_layer_GAP_max
	for k in range(XHCH-1):
		in_layer_GAP_kmax0= tf.where(in_layer_GAP_kmax0 < tf.expand_dims(tf.reduce_max(in_layer_GAP_kmax0, axis=-2), -2),in_layer_GAP_kmax0,GAP_zeros)
		in_layer_GAP_kmax= tf.where(in_layer_GAP_kmax0 < tf.expand_dims(tf.reduce_max(in_layer_GAP_kmax0, axis=-2), -2),GAP_zeros,GAP_ones)
		in_layer_GAP_kmax = tf.reshape(in_layer_GAP_kmax,[batch,1,1,-1,1,channel])
		atten = tf.concat([atten, in_layer_GAP_kmax], -2)

	current_featuremap_kmax = atten*in_layer_all
	current_featuremap_kmax = tf.reduce_sum(current_featuremap_kmax, axis = -3)
	current_featuremap_kmax = tf.reshape(current_featuremap_kmax,[batch,size,size,-1])
	cn_layer =  tf.concat([in_input,current_featuremap_kmax], -1)
	#cn_layer =  tf.concat([in_input,current_featuremap_max], -1)	
	return tf.cast(cn_layer, tf.float32)


T_56 = Template_generator(56, 8, 56/8, 'l1')
T_56 = (np.array(T_56)).astype(np.float16)
T_28 = Template_generator(28, 8, 28/8, 'l1')
T_28 = (np.array(T_28)).astype(np.float16)
T_14 = Template_generator(14, 8, 14/8,'l1')
T_14 = (np.array(T_14)).astype(np.float16)


def PKA2_Net(input_data,n_class,batchss, phase=True,reuse=False):
	with tf.variable_scope('net', reuse=reuse):
		with tf.name_scope('resnet1'):
			with tf.name_scope('conv1_1'):

				input_data1 = tf.layers.conv2d(input_data,64,7,strides=2,kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=sca),padding='same')
				input_data1 = tf.nn.relu(tf.layers.batch_normalization(input_data1, training=phase))
			
			with tf.name_scope('pool1_1'):
				input_data1 = max_pool_3x3(input_data1)
			# with tf.name_scope('res_block1_1'):
				# input_data1 = res_block_down(input_data1,64,phase=phase,NAME='DLN1')                
			
			with tf.name_scope('res_block1_1'):
				for i in range(3):
					input_data1 = res_block(input_data1,64,phase=phase,NAME=str(i) + 'LN1')
				input_data1 = SEBS(T_56, input_data1,56,batchss,64)
				input_data1 = tf.layers.conv2d(input_data1,64,1,strides=1,padding='same')
				
			with tf.name_scope('res_block1_2'):
				input_data1 = res_block_down(input_data1,128,phase=phase,NAME='DLN1')
							
				for i in range(3):
					input_data1 = res_block(input_data1,128,phase=phase,NAME=str(i) + 'LN2')
				input_data1 = SEBS(T_28, input_data1,28,batchss,128)
				input_data1 = tf.layers.conv2d(input_data1,128,1,strides=1,padding='same')


			with tf.name_scope('res_block1_3'):
				input_data1 = res_block_down(input_data1,256,phase=phase,NAME= 'DLN2')
				
				for i in range(5):
					input_data1 = res_block(input_data1,256,phase=phase,NAME=str(i) + 'LN3')
				input_data1 = SEBS(T_14, input_data1,14,batchss,256)
				input_data1 = tf.layers.conv2d(input_data1,256,1,strides=1,padding='same')
			with tf.name_scope('res_block1_4'):
				input_data1 = res_block_down(input_data1,512,phase=phase,NAME= 'DLN3')
			
				for i in range(2):
					input_data1 = res_block(input_data1,512,phase=phase,NAME=str(i) + 'LN4')				

			with tf.name_scope('GAP1'):        
				GAP1 = tf.reduce_mean(input_data1, axis=[1,2])
                
			with tf.name_scope('FC1'):
				fc1 = tf.layers.dense(inputs = GAP1, units = n_class)
		return fc1


#


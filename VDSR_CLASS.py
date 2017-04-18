import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
from MODEL import model
from PSNR import psnr
from TEST import test_VDSR
#from MODEL_FACTORIZED import model_factorized
DATA_PATH = "./data/train_class/"
IMG_SIZE = (41, 41)
BATCH_SIZE = 64
USE_ADAM_OPT = True
if USE_ADAM_OPT:
	BASE_LR = 0.00001
else:
	BASE_LR = 0.1
LR_RATE = 0.1
LR_STEP_SIZE = 50 #epoch
MAX_EPOCH = 500

USE_QUEUE_LOADING = True


parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

noc = 5 # number of classes

TEST_DATA_PATH = "./data/test_class/"

def get_train_list(data_path):
	l = glob.glob(os.path.join(data_path,"*"))
	print len(l)
	l = [f for f in l if re.search("^\d+_\d.mat$", os.path.basename(f))]
	print len(l)
	train_list = []
	class_list = []
	for f in l:
		if os.path.exists(f):
			class_num = int(f[-5:-4])
			for scale in range(2,5):
					temp_path = f[:-4]+("_%d.mat" % scale)
					if os.path.exists(temp_path):
						train_list.append([f, f[:-4]+("_%d.mat" % scale)])
						class_list.append(class_num-1)
						break


#					train_list.append([f, f[:-4]+("_%d_%d.mat" %(class_num, scale) ) )
#			if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat"])
#			if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat"])
#			if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat"])
	print(train_list[0])
	print(train_list[0][1])
	print(len(train_list))
	return train_list, class_list
def get_class_padded_train(train_img, one_hot, noc):

	if noc > 0:
		replicated = np.tile(one_hot, (41, 41, 1))
		padded_img = np.concatenate((train_img,replicated),axis=2)
	else:
		padded_img = train_img

	#print(padded_img.shape)
	return padded_img

def get_image_batch(train_list,offset,batch_size):
	target_list = train_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	cbcr_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	return input_list, gt_list, np.array(cbcr_list)

def get_test_image(test_list, offset, batch_size):
	target_list = test_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	for pair in target_list:
		mat_dict = scipy.io.loadmat(pair[1])
		input_img = None
		if mat_dict.has_key("img_2"): 	input_img = mat_dict["img_2"]
		elif mat_dict.has_key("img_3"): input_img = mat_dict["img_3"]
		elif mat_dict.has_key("img_4"): input_img = mat_dict["img_4"]
		else: continue
		gt_img = scipy.io.loadmat(pair[0])['img_raw']
		input_list.append(input_img[:,:,0])
		gt_list.append(gt_img[:,:,0])
	return input_list, gt_list

if __name__ == '__main__':
	train_list, class_list = get_train_list(DATA_PATH)
	one_hot_classes = np.zeros((len(class_list), 5))
	one_hot_classes[np.arange(len(class_list)), np.asarray(class_list)] = 1

	if not USE_QUEUE_LOADING:
		print "not use queue loading, just sequential loading..."


		### WITHOUT ASYNCHRONOUS DATA LOADING ###

		train_input  	= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
		train_gt  		= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))

		### WITHOUT ASYNCHRONOUS DATA LOADING ###

	else:
		print "use queue loading"


		### WITH ASYNCHRONOUS DATA LOADING ###


		train_input_single  = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1 + noc))
		train_gt_single  	= tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
		q = tf.FIFOQueue(10000, [tf.float32, tf.float32], [[IMG_SIZE[0], IMG_SIZE[1], 1 + noc], [IMG_SIZE[0], IMG_SIZE[1], 1]])
		enqueue_op = q.enqueue([train_input_single, train_gt_single])

		train_input, train_gt	= q.dequeue_many(BATCH_SIZE)

		### WITH ASYNCHRONOUS DATA LOADING ###


	shared_model = tf.make_template('shared_model', model)
	#train_output, weights 	= model(train_input)
	train_output, weights 	= shared_model(train_input)
	loss = tf.reduce_sum((train_output - train_gt)**2)
	for w in weights:
		loss += tf.nn.l2_loss(w)*1e-4

	global_step 	= tf.Variable(0, trainable=False)
	learning_rate 	= tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)

	if USE_ADAM_OPT:
		optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, 0.9)
		opt = optimizer.minimize(loss, global_step=global_step)
	else:
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

		lr = BASE_LR
		BASE_NORM = 0.1
		tvars = tf.trainable_variables()
		gvs = zip(tf.gradients(loss,tvars), tvars)
		#norm = BASE_NORM*BASE_LR/lr
		#capped_gvs = [(tf.clip_by_norm(grad, norm), var) for grad, var in gvs]
		norm = 0.01
		capped_gvs = [(tf.clip_by_norm(grad, norm), var) for grad, var in gvs]
		opt = optimizer.apply_gradients(capped_gvs, global_step=global_step)

	saver = tf.train.Saver(weights, max_to_keep=0)

	def shuffle_in_unison_scary(a, b):
	    rng_state = np.random.get_state()
	    np.random.shuffle(a)
	    np.random.set_state(rng_state)
	    np.random.shuffle(b)

	shuffle_in_unison_scary(train_list, one_hot_classes)


	config = tf.ConfigProto()
	#config.operation_timeout_in_ms=10000

	with tf.Session(config=config) as sess:
		tf.global_variables_initializer().run()

		if model_path:
			print "restore model..."
			saver.restore(sess, model_path)
			print "Done"

		### WITH ASYNCHRONOUS DATA LOADING ###
		"""
		def load_and_enqueue(coord, file_list, idx=0, num_thread=1):
			count = 0;
			length = len(file_list)
			while not coord.should_stop():
				i = (count*num_thread + idx) % length;
				input_img	= scipy.io.loadmat(file_list[i][1])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
				gt_img		= scipy.io.loadmat(file_list[i][0])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
				sess.run(enqueue_op, feed_dict={train_input_single:input_img, train_gt_single:gt_img})
				count+=1
		"""
		def load_and_enqueue(coord, file_list, enqueue_op, train_input_single, train_gt_single, idx=0, num_thread=1):
			count = 0;
			length = len(file_list)
			try:
				while not coord.should_stop():
					i = count % length;
					#i = random.randint(0, length-1)
					input_img	= scipy.io.loadmat(file_list[i][1])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
					gt_img		= scipy.io.loadmat(file_list[i][0])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])

					padded_img = get_class_padded_train(input_img, one_hot_classes[i], 6)
					#input_img_new = np.reshape(padded_img[:,:,1], (41,41,1))
					#gt_img = get_class_padded_train(gt_img, one_hot_classes[i])


					sess.run(enqueue_op, feed_dict={train_input_single:padded_img, train_gt_single:gt_img})
					count+=1
			except Exception as e:
				print "stopping...", idx, e
		"""
		if USE_QUEUE_LOADING:
			# create threads
			coord = tf.train.Coordinator()
			num_thread=1
			for i in range(num_thread):
				length = len(train_list)/num_thread
				t = threading.Thread(target=load_and_enqueue, args=(coord, train_list[i*length:(i+1)*length], i, num_thread))
				t.start()
		"""
		### WITH ASYNCHRONOUS DATA LOADING ###
		threads = []
		def signal_handler(signum,frame):
			#print "stop training, save checkpoint..."
			#saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
			sess.run(q.close(cancel_pending_enqueues=True))
			coord.request_stop()
			coord.join(threads)
			print "Done"
			sys.exit(1)
		original_sigint = signal.getsignal(signal.SIGINT)
		signal.signal(signal.SIGINT, signal_handler)
		best_loss = 10000000
		if USE_QUEUE_LOADING:
			lrr = BASE_LR
			for epoch in xrange(0, MAX_EPOCH):
				if epoch%LR_STEP_SIZE==0:


					train_input_single  = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1 + noc))
					train_gt_single  	= tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
					with tf.container("FIFOQueue"):
						q = tf.FIFOQueue(1000, [tf.float32, tf.float32], [[IMG_SIZE[0], IMG_SIZE[1], 1 + noc], [IMG_SIZE[0], IMG_SIZE[1], 1]])


					enqueue_op = q.enqueue([train_input_single, train_gt_single])

					train_input, train_gt	= q.dequeue_many(BATCH_SIZE)

					### WITH ASYNCHRONOUS DATA LOADING ###
					train_phase = tf.placeholder(tf.bool, name='train_phase')
					train_output, weights 	= shared_model(train_input)
					print("train_gt_shape", train_gt.get_shape())
					print("train_output_shape", train_output.get_shape())
					loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt)))
					for w in weights:
						loss += tf.nn.l2_loss(w)*1e-4

					if USE_ADAM_OPT:
						opt = optimizer.minimize(loss, global_step=global_step)
					else:

						lr = BASE_LR
						BASE_NORM = 0.1
						tvars = tf.trainable_variables()
						gvs = zip(tf.gradients(loss,tvars), tvars)
						#norm = BASE_NORM*BASE_LR/lr
						#capped_gvs = [(tf.clip_by_norm(grad, norm), var) for grad, var in gvs]
						norm = 0.01
						capped_gvs = [(tf.clip_by_norm(grad, norm), var) for grad, var in gvs]
						opt = optimizer.apply_gradients(capped_gvs, global_step=global_step)

				# create threads
				num_thread=20
				print "num thread:" , len(threads)
				del threads[:]
				coord = tf.train.Coordinator()
				print "delete threads..."
				print "num thread:" , len(threads)
				for i in range(num_thread):
					length = len(train_list)/num_thread
					t = threading.Thread(target=load_and_enqueue, args=(coord, train_list[i*length:(i+1)*length],enqueue_op, train_input_single, train_gt_single,  i, num_thread))
					threads.append(t)
					t.start()
				print("adding threads")






				for step in range(len(train_list)//BATCH_SIZE):
					_,l,output,lr, g_step = sess.run([opt, loss, train_output, learning_rate, global_step])
					print "[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr)
					#print "[epoch %2.4f] loss %.4f\t lr %.5f\t norm %.2f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr, norm)
				'''
				if l < best_loss:
					print("Best loss achieved", np.sum(l)/BATCH_SIZE)
					saver.save(sess, "./checkpoints/VDSR_adam_best_train.ckpt" ,global_step=global_step)
					best_loss = l;
				'''


				saver.save(sess, "./checkpoints/VDSR_adam_epoch_%03d.ckpt" % epoch ,global_step=global_step)
				'''
				sess.run(q.close(cancel_pending_enqueues=True))
				print "request stop..."
				coord.request_stop()
				print "join threads..."
				coord.join(threads, stop_grace_period_secs=10)
				tf.Session.reset(target=None, containers=["FIFOQueue"])
				saver.restore(sess, model_path)
				'''


					#lrr = lrr/10
		else:
			for epoch in xrange(0, MAX_EPOCH):
				for step in range(len(train_list)//BATCH_SIZE):
					offset = step*BATCH_SIZE
					input_data, gt_data, cbcr_data = get_image_batch(train_list, offset, BATCH_SIZE)
					feed_dict = {train_input: input_data, train_gt: gt_data}
					_,l,output,lr, g_step = sess.run([opt, loss, train_output, learning_rate, global_step], feed_dict=feed_dict)
					print "[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr)
					del input_data, gt_data, cbcr_data

				saver.save(sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)

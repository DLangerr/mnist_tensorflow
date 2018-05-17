from preprocess_data import preprocess_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import datetime

class Neural_Network(object):
	def __init__(self):
		# X, Y, Y_one_hot, D, N, K = preprocess_data()
		self.desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') 
		self.D = 784
		self.K = 10
		self.phX = tf.placeholder(tf.float32, [None, self.D])
		self.phY = tf.placeholder(tf.float32, [None, self.K])


		self.M1 = 30
		self.M2 = 20

		self.W1 = tf.Variable(tf.random_normal([self.D, self.M1], stddev=0.1))
		self.b1 = tf.Variable(tf.zeros([self.M1]))
		self.W2 = tf.Variable(tf.random_normal([self.M1, self.M2], stddev=0.1))
		self.b2 = tf.Variable(tf.zeros([self.M2]))
		self.W3 = tf.Variable(tf.random_normal([self.M2, self.K], stddev=0.1))
		self.b3 = tf.Variable(tf.zeros([self.K]))

		self.Y_ = self.forward(self.phX)
		self.pred = tf.argmax(self.Y_, 1)

		self.saver = tf.train.Saver()

	def forward(self, X):
		self.Z1 = tf.nn.tanh(tf.matmul(X, self.W1) + self.b1)
		self.Z2 = tf.nn.tanh(tf.matmul(self.Z1, self.W2) + self.b2)
		return tf.matmul(self.Z2, self.W3) + self.b3

	def train(self, epochs=200, lr=0.01, show_fig=False):

		X, Y, Y_one_hot = preprocess_data()

		self.lr = lr

		X_train, X_test = self.train_test_split(X)
		Y_train, Y_test = self.train_test_split(Y)
		Y_train_ohe, Y_test_ohe = self.train_test_split(Y_one_hot)


		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.phY, logits=self.Y_))
		self.train = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

		init = tf.global_variables_initializer()

		

		self.epochs = epochs
		self.cost_array = []
		with tf.Session() as sess:
			sess.run(init)
			for i in range(self.epochs):
				sess.run(self.train, feed_dict={self.phX:X_train, self.phY:Y_train_ohe})
				self.c = np.mean(sess.run(self.cost, feed_dict={self.phX:X_train, self.phY:Y_train_ohe}))
   		
				if i % 10 == 0:
					self.predictions = sess.run(self.pred, feed_dict={self.phX:X_test})
					self.acc = np.mean(self.predictions==Y_test)
					print(f"Iteration {i}. Cost: {self.c}. Accuracy: {self.acc}")
    
				self.cost_array.append(self.c)

			if show_fig:
				plt.plot(self.cost_array)
				plt.show()
			self.save_path = self.saver.save(sess, "/tmp/model.ckpt")
			print("Model saved in path: %s" % self.save_path)

	def train_test_split(self, Z, train_percentage=0.8):
   
		Z_train_size = int(Z.shape[0]*train_percentage)
		Z_test_size = Z.shape[0]-Z_train_size
		Z_train = Z[:Z_train_size]
		Z_test = Z[:Z_test_size]
		return Z_train, Z_test

	def test(self):
		self.images = {}
		print("Reading images . . .")
		for file in os.listdir("test_images"):
			if file.endswith(".png") or file.endswith(".jpg"):
				self.path = os.path.join("test_images", file)
				self.img = Image.open(self.path).convert('L')
				self.pix_img = self.edit_img(self.img)
				self.images[file] = self.pix_img

		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
	
			print("Loading session . . .")
			self.saver.restore(sess, "/tmp/model.ckpt")
			print("Session loaded.")
			os.chdir(self.desktop)
			with open('output.txt', 'a') as f:
				f.write(str(datetime.datetime.now()) + "\n")
				for file, image in self.images.items():
					self.predictions = sess.run(self.pred, feed_dict={self.phX:image})
					self.message = f"Prediction for {file}: {self.predictions}."
					
					print(self.message)
					f.write(self.message)
					f.write("\n")
				f.write("\n\n")


	def edit_img(self, img):
		pix_img = np.array(img)
		pix_img = 255-pix_img
		pix_img = pix_img / 255
		pix_img = pix_img.reshape(1, 784)
		return pix_img


if __name__ == '__main__':
	nn = Neural_Network()
	nn.train()
	nn.test()

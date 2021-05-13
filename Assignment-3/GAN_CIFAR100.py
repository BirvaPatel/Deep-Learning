from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar100 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
 
# define discriminator model
def Discriminator(in_shape=(32,32,3)):
	model = Sequential()
	
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))  
	model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()
	# Compilation of model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 
# define generator model
def Generator(dim_Latent):
	model = Sequential()
	n_nodes = 256 * 4 * 4  # 4x4 image
	model.add(Dense(n_nodes, input_dim=dim_Latent))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) # 8x8
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) # 16x16
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) # 32x32
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	model.summary()
	return model
 
# combine the discriminator and generative model
def Combine(gen_model, dis_model):
	
	dis_model.trainable = False
	model = Sequential()
	model.add(gen_model)
	model.add(dis_model)
	#compile the combined model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
 
# loading the dataset
def load_real_samples():
	
	(x_train, _), (_, _) = load_data()
	X = x_train.astype('float32')
	# scale between [-1,1]
	X = (X - 127.5) / 127.5
	return X
 
# Real samples selection
def Generate_Real_samp(ds, n_samp):
	
	matrix = randint(0, ds.shape[0], n_samp)
	X = ds[matrix]
	y = ones((n_samp, 1))
	return X, y
 
# Create latent points as generator input
def Generate_Latent_Point(dim_Latent, n_samp):
	
	ip_x = randn(dim_Latent * n_samp)
	ip_x = ip_x.reshape(n_samp, dim_Latent)
	return ip_x
 
# Generate fake cases with class labels
def Generate_Fake_samp(gen_model, dim_Latent, n_samp):
	
	ip_x = Generate_Latent_Point(dim_Latent, n_samp)
	X = gen_model.predict(ip_x)
	y = zeros((n_samp, 1))
	return X, y
 
# Generate and save the images
def save_plot(cases, e, n=7):
	# scale between [0,1]
	cases = (cases + 1) / 2.0
	# plot images
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(cases[i])
	f_name = 'Generated images/generated_plot_e%03d.png' % (e+1)
	pyplot.savefig(f_name)
	pyplot.close()
 
# Discriminator evaluation 
def summarize_performance(e, gen_model, dis_model, ds, dim_Latent, n_samp=150):
	
	Real_x, Real_y = Generate_Real_samp(ds, n_samp)
	_, Real_accuracy = dis_model.evaluate(Real_x, Real_y, verbose=0)
	
	Fake_x, Fake_y = Generate_Fake_samp(gen_model, dim_Latent, n_samp)
	_, Fake_accuracy = dis_model.evaluate(Fake_x, Fake_y, verbose=0)
	
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (Real_accuracy*100, Fake_accuracy*100))
	
	save_plot(Fake_x, e)
	f_name = 'Generated model/generator_model_%03d.h5' % (e+1)
	gen_model.save(f_name)
 
# Train the generator and discriminator
def train(gen_model, dis_model, gan_model, ds, dim_Latent, n_es=200, n_bat=128):
	batch_p_e = int(ds.shape[0] / n_bat)
	half_bat = int(n_bat / 2)
	for i in range(n_es):
		for j in range(batch_p_e):
			Real_x, Real_y = Generate_Real_samp(ds, half_bat)
			dis_loss1, _ = dis_model.train_on_batch(Real_x, Real_y)
			Fake_x, Fake_y = Generate_Fake_samp(gen_model, dim_Latent, half_bat)
			dis_loss2, _ = dis_model.train_on_batch(Fake_x, Fake_y)
			Gan_x = Generate_Latent_Point(dim_Latent, n_bat)
			Gan_y = ones((n_bat, 1))
			gen_loss = gan_model.train_on_batch(Gan_x, Gan_y)
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, batch_p_e, dis_loss1, dis_loss2, gen_loss))
		if (i+1) % 10 == 0:
			summarize_performance(i, gen_model, dis_model, ds, dim_Latent)
 
# Latent space size
dim_Latent = 100
dis_model = Discriminator()
gen_model = Generator(dim_Latent)
gan_model = Combine(gen_model, dis_model)
ds = load_real_samples()
train(gen_model, dis_model, gan_model, ds, dim_Latent)


# Sample of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

# generate points as input for the generator in latent space 
def Generate_Latent_Point(dim_Latent, n_samp):
	ip_x = randn(dim_Latent * n_samp)
	ip_x = ip_x.reshape(n_samp, dim_Latent)
	return ip_x

# plot the generated images
def create_plot(cases, n):
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(cases[i, :, :])
	pyplot.show()

# load model
model = load_model('Generated model/generator_model_200.h5')
latent_points = Generate_Latent_Point(100, 100)
X = model.predict(latent_points)
X = (X + 1) / 2.0
create_plot(X, 10)




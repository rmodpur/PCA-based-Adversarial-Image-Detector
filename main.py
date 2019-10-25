# If you want to run it on google Colab
# !pip install https://github.com/bethgelab/foolbox/archive/master.zip
# !pip install randomgen

from foolbox import zoo
url = "https://github.com/bethgelab/cifar10_challenge.git"
model = zoo.get_model(url)

import numpy as np
import keras
import foolbox
import randomgen
from keras.datasets import cifar10
from sklearn.decomposition import PCA

''' Loading cifar-10 dataset'''

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train=np.reshape(y_train,(50000))
y_test=np.reshape(y_test,(10000))


''' Performing PCA on training data to get all principal components'''

data = np.reshape(x_train,(50000,3072))
pca=PCA(n_components=3072).fit(data)

''' Reading input images'''

test = np.load('adversarial_images.npy')

# dimensions of each image is 32*32*3 = 3072
dimensions = 3072
no_of_image = len(test)/dimensions
test = np.reshape(test,(no_of_image, dimensions))

''' Getting predictions from classifier for input images'''

pred = model.forward(np.reshape(test,(no_of_image,32,32,3)))
preds_a = np.array([np.argmax(pred[i]) for i in range(no_of_image)])


''' Transforming input images along principal components'''

trans_x = pca.transform(test)


''' Detecting whether input images are adversarial or not

    n_components : number of least significant coefficients to perturb
    n_sample     : number of samples to generate
    perturbation : a perturbation sampled from std normal for n_components
    inv_x        : inverse transformed image after adding perturbation
    preds_y      : classifier predctions for updated images
    preds_a      : classifier predctions for updated images
    result       : output will be stored in result array at cooresponding indices as follows:
                    1 for adversarial images
                    0 for non-adversarial images 
'''

n_components = 1000
n_sample = 25
rng = randomgen.RandomGenerator()

result = np.zeros(no_of_image, dtype = int)

for k in range(n_sample):
  perturbation = rng.standard_normal(size=(n_components,), dtype=trans_x.dtype)
  new_x = np.copy(trans_x)
  for i in range(no_of_image):
    new_x[i][(3072-n_components):] += 10*perturbation

  inv_x = pca.inverse_transform(new_x)
  invx = np.reshape(inv_x,((no_of_image,32,32,3))
                    
  pred = model.forward(invx)
  preds_y = np.array([np.argmax(pred[i]) for i in range(n-s)])
                    
  res = np.not_equal(preds_y, preds_a)
  result = np.logical_or(res,result)
                    
''' Printing total number of adversarial images out of all input images'''
print(np.sum(result))

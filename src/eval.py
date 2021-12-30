#import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from math import log10, sqrt
import glob
'''evaluate(orig_path, pred_path)
    Arguments:
    orig_path: original folder path
    pred_path: colorised folder path
    outputs:
    the average psnr and average ssim values
'''
def PSNR(imageA, imageB):
    
	mse = np.mean((imageA - imageB) ** 2)
	if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB,multichannel=True)
	p = PSNR(imageA, imageB) 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f, PSNR:%.2f" % (m, s, p))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB)
	plt.axis("off")
	# show the images
	plt.show()
	return m, s, p

def last_4chars(x):
    return(x[-24:])

def evaluate(orig_path, pred_path):
    orig_files = glob.glob(orig_path+'/*')
    pred_files = glob.glob(pred_path+'/*')
    #pred_files = sorted(pred_files, key = last_4chars)   

    mse = []
    psnr = []
    ssim = []
    for x,y in zip(orig_files[:], pred_files[:]):
        img = cv2.imread(x)
        orig = cv2.resize(img, (256, 256))
        pred = cv2.imread(y)
        pred = cv2.resize(pred, (256,256))
        
        m, s, p = compare_images(orig, pred, "Original vs. Color")
        #p = PSNR(x, y)
        mse.append(m)
        psnr.append(p)
        ssim.append(s)
    #mse = np.mean(mse)
    avg_psnr = np.mean(psnr)
    ssim = np.mean(ssim)
    print(psnr)
    print(avg_psnr)
    print(ssim)
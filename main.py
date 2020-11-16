# Jessica Nono

from PIL import Image
import numpy as np
import cv2
import torch
import PennFudanDataset
def convolve(image,kernel, bias):
   m, n = kernel.shape
   if (m == n):
       y, x = image.shape
       y = y - m + 1
       x = x - m + 1
       new_image = np.zeros((y, x))
       for i in range(y):
           for j in range(x):
               new_image[i][j] = np.sum(image[i:i+m, j:j+m]* kernel) + bias
   return new_image

def readImage(path):
    im = np.array((Image.open(path)).convert('RGB'))
    return im

def rgbTogray(imag):
    #matrix product @ Y=0.2126R+0.7152G+0.0722B
    g = imag @ np.array([0.299, 0.7512, 0.0722]) # gray scale now
    im = Image.fromarray(g)
    im.convert('RGB').save('data/gray.jpg')
    im.show()
    return g

def sobel(imag):
    #matrix product @  wiki formula
    filter1  =  np.array([[-1, 0, 1], [-2, 0, 2], [-1,-0, 1]])
    filter2 = np.array([[-1, -2, -1], [0, 0,0], [1,2,1]])
    new_image_x = convolve(imag, filter1,2)
    print(new_image_x)
    new_image_y = convolve(imag, filter2,2)
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    im = Image.fromarray(gradient_magnitude)
    im.convert('RGB').save('data/edgesobel.jpg')
    im.show()
    theta = np.arctan2(new_image_x, new_image_y)
    return (gradient_magnitude, theta)

#Non-Maximum Suppression

#Ideally, the final image should have thin edges. Thus, we must perform non-maximum suppression to thin out the edges.

#The principle is simple: the algorithm goes through all the points on the gradient intensity matrix and finds the pixels with the maximum value in the edge directions.

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    im = Image.fromarray(img)
    im.convert('RGB').save('data/nondupp.jpg')
    im.show()
    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    im = Image.fromarray(res)
    im.convert('RGB').save('data/threo.jpg')
    im.show()

    return (res, weak, strong)


#Edge Tracking by Hysteresis

#Based on the threshold results,
# the hysteresis consists of transforming weak pixels into strong ones, if and only if at least one of the pixels around the
def hysteresis(img, weak, strong=0):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    im = Image.fromarray(img)
    im.convert('RGB').save('data/hyste.jpg')
    im.show()
    return img

def contourf(image):
    copy = cv2.copyMakeBorder(image, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    im = Image.fromarray(copy)
    im.convert('RGB').save('data/copy.jpg')
    im = cv2.imread('data/copy.jpg')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    _red = (0, 0, 255);  # Red for external contours
    _green = (0, 255, 0);  # Gren internal contours
    levels = 2  # 1 contours drawn, 2 internal contours as well, 3 ...
    cv2.drawContours(copy, contours, 2, _green, 2)

    im = Image.fromarray(copy)
    im.convert('RGB').save('data/contours.jpg')
    im.show()

    return copy
if __name__ == '__main__':
   im =   readImage('data/test1.jpg')

   im2 = readImage('data/mario.png')
   gray = rgbTogray(im)
   gradientMat, thetaMat = sobel(gray)
   nonmaxsupr = non_max_suppression(gradientMat, thetaMat)
   res, weak, strong = threshold(nonmaxsupr)
   img_final = hysteresis(res, weak,0)

  # mask = np.zeros(gray.shape[:2], np.uint8)
   #mask[100:300, 100:400] = 255
   #masked_img = cv2.bitwise_and(im, im, mask=mask)

   # Calculate histogram with mask and without mask
   # Check third argument for mask
   #hist_full = cv2.calcHist([im], [0], None, [256], [0, 256])
  # hist_mask = cv2.calcHist([im], [0], mask, [256], [0, 256])

   #plt.subplot(221), plt.imshow(im , 'gray')
   #plt.subplot(222), plt.imshow(mask, 'gray')
   #plt.subplot(223), plt.imshow(masked_img, 'gray')
   #plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
   #plt.xlim([0, 256])

  # plt.show()


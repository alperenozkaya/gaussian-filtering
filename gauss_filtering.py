from PIL import *
from PIL import Image
import numpy as np

def main():
    img = readPILimg()
    arr = PIL2np(img)

    my_filter = np.array([[-1, 0, 1 ], [-1, 0, 1 ], [-1, 0, 1]])
    g_filter = create_Gaussian_filter(1.5)
    im_out = convolve(arr,g_filter)
    new_img = np2PIL(im_out)
    new_img.show()

def readPILimg():
    img = Image.open('images/bird.png')
    #img.show()
    img_gray = color2gray(img)
    img_gray.show()
    #img_gray.save('/Users/gokmen/Dropbox/vision-python/images/brick-house-gs','png')
    #new_img = img.resize((256,256))
    #new_img.show()
    return img_gray

def color2gray(img):
    img_gray = img.convert('L')
    return img_gray

def PIL2np(img):
    nrows = img.size[0]
    ncols = img.size[1]
    print("nrows, ncols : ", nrows,ncols)
    imgarray = np.array(img.convert("L"))
    return imgarray

def np2PIL(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(np.uint8(im))
    return img

def create_Gaussian_filter(sigma):
    nrow = int(2 * sigma + 1)
    if nrow % 2 == 0:
        nrow += 1
    ncol = nrow
    half_row = int((nrow - 1) / 2)
    half_col = int((ncol - 1) / 2)
    gauss_filter = np.zeros(shape = (nrow, ncol))
    sum = 0.
    for i in range(-half_row, half_row + 1):
        for j in range(-half_col, half_col + 1):
            gauss_filter[i + half_row][j + half_col] =  (np.exp(-(i * i + j * j ) / (2.0 * sigma * sigma))) / (2. * 3.1415 * sigma * sigma)
            sum += gauss_filter[i + half_row][j + half_col]
    print("gauss filter : ")
    print(np.array(gauss_filter) / sum)
    gauss_filter = np.array(gauss_filter) / sum
    return gauss_filter

def convolve(im,filter):
    (nrows, ncols) = im.shape
    (k1,k2) = filter.shape
    k1h = int((k1 -1) / 2)
    k2h = int((k2 -1) / 2)
    im_out = np.zeros(shape = im.shape)
    print("image size , filter size ", nrows, ncols, k1, k2)
    for i in range(k1h, nrows - k1h):
        for j in range(k2h, ncols - k2h):
            sum = 0.
            for l in range(-k1h, k1h + 1):
                for m in range(-k2h, k2h + 1):
                    sum += im[i - l][j - m] * filter[l + k1h][m + k2h]
            im_out[i][j] = sum
    return im_out
dfdsfs

if __name__=='__main__':
    main()

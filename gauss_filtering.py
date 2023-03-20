from PIL import *
from PIL import Image
import numpy as np


def main():
    img = readPILimg()
    arr = PIL2np(img)

    # my_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # g_filter = create_Gaussian_filter(1.5)
    # im_out = convolve(arr, g_filter)
    # new_img = np2PIL(im_out)
    # new_img.show()

    # Task 1 Show the results with different sigma values. Interpret the results.

    sigma_values = [3, 4, 5]

    for sigma in sigma_values:
        g_filter = create_Gaussian_filter(sigma)
        im_out = convolve(arr, g_filter)
        new_img = np2PIL(im_out)
        new_img.save(f'output_sigma_{sigma}.png')
        new_img.show()

    # Task 2: Using your program, verify that G(x, sigma1) * G(x,sigma2) = G(x, sigma3) where
    # sigma1**2 + sigma2**2 = sigma3**2

    verify_convolution(3, 4)
    # verify_convolution(1, 2)

    # Task 3: Obtain a Gaussion pyramid of a given image using original image and 5 levels
    # of the pyramid. Describe how you obtain the pyramid, Show the pyramid.
    # Discuss where the Gaussian pyramids can be used.

    pyramid = create_gaussian_pyramid(img, sigma=1)
    # display_images(pyramid)
    display_pyramid(pyramid)

    # Task 4: Take the difference between two convolved images with different sigma
    # values. Interpret the results.

    difference_convolved_images(1.5, 2)


def difference_convolved_images(sigma1, sigma2):
    img = readPILimg()
    arr = PIL2np(img)

    gaussian_filter_sigma1 = create_Gaussian_filter(sigma1)
    gaussian_filter_sigma2 = create_Gaussian_filter(sigma2)

    im_out_sigma1 = convolve(arr, gaussian_filter_sigma1)
    im_out_sigma2 = convolve(arr, gaussian_filter_sigma2)

    img_sigma1 = np2PIL(im_out_sigma1)
    img_sigma2 = np2PIL(im_out_sigma2)

    difference = (im_out_sigma1 - im_out_sigma2)
    img_difference = np2PIL(difference)
    img_difference.show()
    #img_difference.save('difference_image.png')

    canvas_width = len(arr[0]) * 3
    canvas_height = len(arr[1])

    canvas = Image.new('L', (canvas_width, canvas_height))

    x_offset = 0

    canvas.paste(img_sigma1, (x_offset, 0))
    x_offset += len(arr[0])

    canvas.paste(img_sigma2, (x_offset, 0))
    x_offset += len(arr[0])

    canvas.paste(img_difference, (x_offset, 0))
    x_offset += len(arr[0])

    canvas.show()
    canvas.save('difference.png')


def create_gaussian_pyramid(img, sigma):
    pyramid = [img]

    g_filter = create_Gaussian_filter(sigma)

    for i in range(0, 4):
        arr = PIL2np(img)
        im_out = convolve(arr, g_filter)
        img = np2PIL(im_out)
        img = img.resize((img.size[0] // 2, img.size[1] // 2))
        pyramid.append(img)
    return pyramid


def display_images(pyramid):
    for img in pyramid:
        img.show()


def display_pyramid(pyramid):
    canvas_width = sum([level.size[0] for level in pyramid])
    canvas_height = max([level.size[1] for level in pyramid])

    canvas = Image.new('L', (canvas_width, canvas_height))

    x_offset = 0
    for level in pyramid:
        canvas.paste(level, (x_offset, 0))
        x_offset += level.size[0]

    canvas.show()
    canvas.save('gaussian_pyramid.png')


def verify_convolution(sigma1, sigma2):
    sigma3 = np.sqrt(sigma1 ** 2 + sigma2 ** 2)

    img = readPILimg()
    arr = PIL2np(img)

    # gaussian filters are created
    gaussian_filter_sigma1 = create_Gaussian_filter(sigma1)
    gaussian_filter_sigma2 = create_Gaussian_filter(sigma2)
    gaussian_filter_sigma3 = create_Gaussian_filter(sigma3)

    # filtered images
    im_out_sigma1 = convolve(arr, gaussian_filter_sigma1)
    im_out_sigma2 = convolve(arr, gaussian_filter_sigma2)
    im_out_sigma3 = convolve(arr, gaussian_filter_sigma3)

    # G(x, sigma1) * G(x,sigma2) (sequential convolution)
    sigma1_sigma2_conv = convolve(im_out_sigma1, gaussian_filter_sigma2)

    new_img_sigma3 = np2PIL(im_out_sigma3)  # G(x, sigma3)
    new_img_sigma3.save(f'output_sigma3_task2.png')
    new_img_sigma3.show()

    new_img_sigma1_2 = np2PIL(sigma1_sigma2_conv)  # G(x, sigma1) * G(x,sigma2)
    new_img_sigma1_2.save(f'output_sigma1_sigma2_task2.png')
    new_img_sigma1_2.show()



def readPILimg():
    img = Image.open('images/bird.png')
    #img.show()
    img_gray = color2gray(img)
    #img_gray.show()
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
    print("nrows, ncols : ", nrows, ncols)
    imgarray = np.array(img.convert("L"))
    return imgarray


def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


def create_Gaussian_filter(sigma):
    nrow = int(2 * sigma + 1)
    # nrow = int(4 * sigma + 1) changed from 2 to 4
    if nrow % 2 == 0:
        nrow += 1
    ncol = nrow
    half_row = int((nrow - 1) / 2)
    half_col = int((ncol - 1) / 2)
    gauss_filter = np.zeros(shape=(nrow, ncol))
    sum = 0.
    for i in range(-half_row, half_row + 1):
        for j in range(-half_col, half_col + 1):
            gauss_filter[i + half_row][j + half_col] = (np.exp(-(i * i + j * j) / (2.0 * sigma * sigma))) / (2. * 3.1415 * sigma * sigma)
            sum += gauss_filter[i + half_row][j + half_col]
    print("gauss filter : ")
    print(np.array(gauss_filter) / sum)
    gauss_filter = np.array(gauss_filter) / sum
    return gauss_filter


def convolve(im, filter):
    (nrows, ncols) = im.shape
    (k1, k2) = filter.shape
    k1h = int((k1 - 1) / 2)
    k2h = int((k2 - 1) / 2)
    im_out = np.zeros(shape=im.shape)
    print("image size , filter size ", nrows, ncols, k1, k2)
    for i in range(k1h, nrows - k1h):
        for j in range(k2h, ncols - k2h):
            sum = 0.
            for l in range(-k1h, k1h + 1):
                for m in range(-k2h, k2h + 1):
                    sum += im[i - l][j - m] * filter[l + k1h][m + k2h]
            im_out[i][j] = sum
    return im_out


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.signal import convolve2d, convolve

MAX_GRAY_LEVEL_VAL = 255
RGB_REP = 2
GRAYSCALE_REP = 1
RGB_SHAPE = 3
CONVOLUTION_WITH_X = np.array([[-1, 0, 1]])
CONVOLUTION_WITH_Y = np.transpose(CONVOLUTION_WITH_X)


def read_image(fileame, representation):
    """ reads an image file and converts it into a given representation
        representation -  is a code, either 1 or 2 defining whether the
        output should be a grayscale image (1) or an RGB image (2)"""

    if representation == GRAYSCALE_REP:
        return imread(fileame, True).astype(np.float64) / MAX_GRAY_LEVEL_VAL
    elif representation == RGB_REP:
        return imread(fileame).astype(np.float64) / MAX_GRAY_LEVEL_VAL


def Dft_matrix(size):
    """ return dft matrix with received size"""
    power = ((-2 * np.pi * 1j) / size) * np.arange(size)
    power = power * np.arange(size)[:, np.newaxis]
    dft_matrix = np.exp(power)
    return dft_matrix


def trans_sig(signal, trans_matrix):
    try:
        return np.dot(trans_matrix, signal)
    except ValueError:
        return np.dot(signal, trans_matrix)


def DFT(signal):
    """ transform a 1D discrete signal to its Fourier representation
    signal is an array of dtype float64 with shape (N,1)"""
    N = signal.shape[0]
    return trans_sig(signal, Dft_matrix(N))


def IDFT(fourier_signal):
    """transform  Fourier representation to its 1D discrete signal
    fourier_signal is an array of dtype complex128 with shape (N,1)"""
    N = fourier_signal.shape[0]
    idft_matrix = np.linalg.inv(Dft_matrix(N))
    return trans_sig(fourier_signal, idft_matrix)


def DFT2(image):
    """transform a 2D discrete signal to its Fourier representation """
    return np.transpose(DFT(np.transpose(DFT(image))))


def IDFT2(fourier_image):
    """transform  Fourier representation to its 2D discrete signa """
    return np.transpose(IDFT(np.transpose(IDFT(fourier_image))))


def conv_der(im):
    """ computes the magnitude of image derivatives."""
    dX = convolve2d(im, CONVOLUTION_WITH_X, mode="same")
    dY = convolve2d(im, CONVOLUTION_WITH_Y, mode="same")
    return np.sqrt(np.abs(dX) ** 2 + np.abs(dY) ** 2)


def fourier_der(im):
    """ computes the magnitude of image derivatives using Fourier transform."""

    fourier_trans = DFT2(im)
    shifted_im = np.fft.fftshift(fourier_trans)

    # compute the x derivative of f
    nX = im.shape[0]
    uArr = np.arange(-nX // 2, nX // 2)
    DFTu = np.transpose(np.transpose(shifted_im) * uArr)
    dX = IDFT2(DFTu)

    # compute the y derivative of f
    nY = im.shape[1]
    vArr = np.arange(-nY // 2, nY // 2)
    DFTv = shifted_im * vArr
    dY = IDFT2(DFTv)

    return np.sqrt(np.abs(dX) ** 2 + np.abs(dY) ** 2)


def get_2D_gaussian(n):
    """ return 2D gaussian kernel g"""
    if n == 1:
        return np.array([[1]])

    gaussian_kernel = np.float64(np.array([1, 1]))

    for i in range(n - 2):
        gaussian_kernel = convolve(gaussian_kernel, np.array([1, 1]))

    gaussian_kernel = gaussian_kernel.reshape(gaussian_kernel.size, 1)
    gaussian_kernel2D = convolve2d(np.transpose(gaussian_kernel), gaussian_kernel)
    return gaussian_kernel2D / np.sum(gaussian_kernel2D)


def blur_spatial(im, kernel_size):
    """ performs image blurring using 2D convolution between the image f and a gaussian kernel g"""
    gaus_matrix = get_2D_gaussian(kernel_size)
    return convolve2d(im, gaus_matrix, mode="same")


def blur_fourier(im, kernel_size):
    """ performs image blurring with gaussian kernel in Fourier space."""
    x = im.shape[0]
    y = im.shape[1]
    gaus_matrix = get_2D_gaussian(kernel_size)
    new_gaus = np.zeros(im.shape)
    new_gaus[int(np.floor(x / 2) - np.floor(kernel_size / 2)):int(np.floor(x / 2) + np.floor(kernel_size / 2)) + 1,
    int(np.floor(y / 2) - np.floor(kernel_size / 2)):int(np.floor(y / 2) + np.floor(kernel_size / 2)) + 1] = gaus_matrix

    return np.fft.ifftshift(IDFT2(DFT2(im) * DFT2(new_gaus)))

# DFT_CONV
Image Processing - Implementing Discrete Fourier Transform, Performing image derivative, and image blurring.


### original image

![orig_im](https://user-images.githubusercontent.com/33607030/53680348-f9a31200-3ce2-11e9-8e1a-0c7d8a6a82ad.png)


### The derivative image

#### Using convolution with [1, 0, −1]

![conv_der](https://user-images.githubusercontent.com/33607030/53680346-f9a31200-3ce2-11e9-9386-842a0e60510f.png)

#### Using Fourier transform

![f_der](https://user-images.githubusercontent.com/33607030/53680347-f9a31200-3ce2-11e9-9adc-f6ecd66ed185.png)


### Image Blurring

#### using a convolution with gaussian kernel g=3 in image space

![blur_conv_3](https://user-images.githubusercontent.com/33607030/53680351-fa3ba880-3ce2-11e9-8062-e79de905e013.png)

#### using a point-wise operation between the fourier image F with the same gaussian kernel, but in fourier space G=3

![blur_f_3](https://user-images.githubusercontent.com/33607030/53680350-f9a31200-3ce2-11e9-9d68-d2c6c1affc22.png)



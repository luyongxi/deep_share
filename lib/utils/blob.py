#------------------------------
# Written by Yongxi Lu
#------------------------------


""" Blob helper functions. """

import numpy as np
import cv2

def im_list_to_blob(filelist, pixel_means, scale):
    """ Load a lits of images, convert them into a network input"""
    
    # file list to image list
    imgs = [cv2.imread(fn) for fn in filelist]

    num_images = len(imgs)
    blob = np.zeros((num_images, scale, scale, 3), dtype=np.float32)
    for i in xrange(num_images):
        im = prep_im_for_blob(imgs[i], pixel_means, scale)
        blob[i, 0:scale, 0:scale, :] = im
    # permute channel to (batch_size, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, scale):
    """ Mean subtract and scale an image """

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im = cv2.resize(im, dsize=(scale,scale), interpolation=cv2.INTER_LINEAR)

    return im

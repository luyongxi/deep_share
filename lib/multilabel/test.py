# ---------------------------------
# Written by Yongxi Lu
# ---------------------------------

"Test a multi-label classification model on a given network"

import numpy as np
from utils.blob import im_list_to_blob
from utils.timer import Timer
from utils.config import cfg

def test_model(net, imdb, cls_idx):
    """ Test a model on imdb. """

    if cls_idx is None:
        cls_idx = np.arange(imdb.num_classes)
    num_classes = len(cls_idx)
    num_images = imdb.num_images

    print '---------------------------------------------------------------'
    print '!!! Test model on the "{}" dataset'.format(imdb.name)
    print '!!! The dataset has {} images.'.format(imdb.num_images)

    # iterate over images, collect error vectors
    err = np.zeros((num_images, num_classes)) # in {0,1} format
    timer = Timer()
    for i in xrange(num_images):
        # prepare blobs 
        label_name = "score"
        fn = imdb.image_path_at(i)
        data = im_list_to_blob([fn], cfg.PIXEL_MEANS, cfg.SCALE)
        net.blobs['data'].reshape(*(data.shape))
        # forward the network
        timer.tic()
        blobs_out = net.forward(data=data.astype(np.float32, copy=False))
        timer.toc()
        # get results
        scores = blobs_out[label_name]
        # evaluate the scores
        err[i,:] = imdb.evaluate(scores, np.array([i]), cls_idx)
        # print infos
        print 'Image {}/{} ::: speed: {:.3f}s per image.'.format(i, num_images, timer.average_time)

    # print out basic dataset information
    print '!!! The average run time is {} per image.'.format(timer.average_time)

    # get error for each class
    class_names = imdb.classes
    mean_err = err.mean(axis=0)
    for i in xrange(len(cls_idx)):
        print '!!! Error rate for class {} is: {}'.\
            format(class_names[cls_idx[i]], mean_err[i])

    print '!!! The average error rate is {}.'.format(mean_err.mean())    
    print '---------------------------------------------------------------'

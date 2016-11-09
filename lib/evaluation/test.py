# ---------------------------------
# Written by Yongxi Lu
# ---------------------------------

"Test a on a given network and a given dataset"

import numpy as np
from utils.blob import im_list_to_blob
from utils.timer import Timer
from utils.config import cfg
import cPickle

import os.path as osp

def test_cls_topk(net, imdb, cls_idx, k):
    """ Test a model on imdb and evaluate the top-k accuracy metric  """

    if cls_idx is None:
        cls_idx = np.arange(imdb.num_classes)

    num_classes = len(cls_idx)
    assert k<=num_classes, 'k={} should be less than or equal to num_classes={}'.\
        format(k, num_classes)
    num_images = imdb.num_images

    # iterate over images, collect error vectors
    found = np.zeros((num_images, num_classes)) # in {0,1} format
    timer = Timer()
    for i in xrange(num_images):
        # prepare blobs 
        label_name = "prob"
        fn = imdb.image_path_at(i)
        data = im_list_to_blob([fn], cfg.PIXEL_MEANS, cfg.SCALE)
        net.blobs['data'].reshape(*(data.shape))
        # forward the network
        timer.tic()
        blobs_out = net.forward(data=data.astype(np.float32, copy=False))
        timer.toc()
        # get results
        scores = blobs_out[label_name]
        # find recall of top-k attributes
        top_classes = np.argsort(-scores)[0, :k]
        pos_classes = np.where(imdb.gtdb['attr'][i, cls_idx] == 1)[0]

        found_classes =  [idx for idx in pos_classes if idx in top_classes]
        found[i, found_classes] = 1.0

        # print infos
        print 'Image {}/{} ::: speed: {:.3f}s per image.'.format(i, num_images, timer.average_time)

    # print out basic dataset information
    print '---------------------------------------------------------------'
    print '!!! Summary of results.'
    print '!!! Test model on the "{}" dataset'.format(imdb.name)
    print '!!! The dataset has {} images.'.format(imdb.num_images)
    print '!!! On average, there are {} active attribute classese per image'.format(np.mean(np.sum(imdb.gtdb['attr'][:, cls_idx], axis=1)))
    print '!!! The average run time is {} per image.'.format(timer.average_time)

    # get error for each class
    class_names = imdb.classes
    recall = np.nansum(found, axis=0)/np.sum(imdb.gtdb['attr'][:, cls_idx], axis=0)
    for i in xrange(len(cls_idx)):
        print '!!! Top {} recall rate for class {} is: {}'.\
            format(k, class_names[cls_idx[i]], recall[i])

    print '!!! The top-{} recall rate is {}.'.format(k, recall.mean())
    print '---------------------------------------------------------------'


def test_cls_error(net, imdb, cls_idx):
    """ Test a model on imdb and evaluate the error rate. """

    if cls_idx is None:
        cls_idx = np.arange(imdb.num_classes)

    num_classes = len(cls_idx)
    num_images = imdb.num_images

    # iterate over images, collect error vectors
    err = np.zeros((num_images, num_classes)) # in {0,1} format
    timer = Timer()
    for i in xrange(num_images):
        # prepare blobs 
        label_name = "prob"
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
    print '---------------------------------------------------------------'
    print '!!! Summary of results.'
    print '!!! Test model on the "{}" dataset'.format(imdb.name)
    print '!!! The dataset has {} images.'.format(imdb.num_images)
    print '!!! The average run time is {} per image.'.format(timer.average_time)

    # get error for each class
    class_names = imdb.classes
    mean_err = np.nanmean(err, axis=0)
    for i in xrange(len(cls_idx)):
        print '!!! Error rate for class {} is: {}'.\
            format(class_names[cls_idx[i]], mean_err[i])

    print '!!! The average error rate is {}.'.format(mean_err.mean())
    print '---------------------------------------------------------------'

def save_softlabels(net, image_list, score_file, labeler):
    """ Save the labels over a set of images of the given class to a file """

    num_images = len(image_list)
    # iterate over images, collect error vectors
    scores = -1.0 * np.ones((num_images, ), dtype=np.float32)
    # decode labler
    score_name = labeler[1]
    score_idx = labeler[2]
    timer = Timer()
    for i in xrange(num_images):
        # prepare blobs
        fn = image_list[i]
        # print 'Image {}/{}: {}'.format(i, num_images, fn)
        data = im_list_to_blob([fn], cfg.PIXEL_MEANS, cfg.SCALE)
        net.blobs['data'].reshape(*(data.shape))
        # forward the network
        timer.tic()
        blobs_out = net.forward(data=data.astype(np.float32, copy=False))
        timer.toc()
        # get results
        scores[i] = blobs_out[score_name][:, score_idx]
        # print infos
        print 'Image {}/{} ::: speed: {:.3f}s per image.'.format(i, num_images, timer.average_time)

    # print out basic dataset information
    print '---------------------------------------------------------------'
    with open(score_file, 'wb') as fid:
        cPickle.dump(scores, fid, cPickle.HIGHEST_PROTOCOL)
        print '!!! The scores are saved to {}.'.format(score_file)

def eval_and_save(net, imdb, cls_idx):
    """ Evaluate the network, and save the scores to a given destination """

    if cls_idx is None:
        cls_idx = np.arange(imdb.num_classes)

    num_classes = len(cls_idx)
    num_images = imdb.num_images

    # iterate over images, collect error vectors
    scores = np.zeros((num_images, num_classes)) # in {0,1} format
    timer = Timer()
    for i in xrange(num_images):
        # prepare blobs 
        label_name = "prob"
        fn = imdb.image_path_at(i)
        data = im_list_to_blob([fn], cfg.PIXEL_MEANS, cfg.SCALE)
        net.blobs['data'].reshape(*(data.shape))
        # forward the network
        timer.tic()
        blobs_out = net.forward(data=data.astype(np.float32, copy=False))
        timer.toc()
        # get results
        scores[i, cls_idx] = blobs_out[label_name]
        # print infos
        print 'Image {}/{} ::: speed: {:.3f}s per image.'.format(i, num_images, timer.average_time)

    # save scores as a pkl file
    score_fn = osp.join(imdb.data_path, imdb.name+'.pkl')
    with open(score_fn, 'wb') as fid:
        cPickle.dump(scores, fid, cPickle.HIGHEST_PROTOCOL)
        print '!!! The scores are saved to {}.'.format(score_fn)


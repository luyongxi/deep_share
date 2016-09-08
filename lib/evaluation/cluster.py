# Written by Yongxi Lu

""" Experiment and visualize clustering of class labels """

import numpy as np
from utils.blob import im_list_to_blob
from utils.timer import Timer
from utils.config import cfg
from sklearn.cluster import SpectralClustering

def _error2Aff(label):
    """ Given an binary label matrix, convert it to correlation matrix.
        label: N by C matrix in {0,1}. 
        cm: C by C matrix in [-1,1].
    """
     # covert to {-1, 1} format
    label = 2.0*label-1.0
    cm = np.dot(label.transpose(), label)
    N = label.shape[0]
    cm = cm/N

    return (cm+1.0)/2.0

def MultiLabel_ECM_cluster(net, k, imdb, cls_idx=None, reverse=False):
    """ Get Multi-label Label Correlation Matrix (LCM) """
    # class list is an ordered list of class index (onto the given dataset)
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
        print 'Image {}/{} ::: speed: {:.3f}s /iter'.format(i, num_images, timer.average_time)

    # get error correlation matrix
    aff = _error2Aff(err)
    if reverse:
        aff = 1.0-aff
    # perform clustering
    return _clusterAffinity(aff, k, imdb, cls_idx)

def _clusterAffinity(aff, k, imdb, cls_idx):
    """ Cluster error correlation matrix using spectral clustering into k cluster, 
        show the class labels in each cluster. 
    """
    # clustering model
    spectral = SpectralClustering(n_clusters=k,
                                  eigen_solver='arpack',
                                  affinity="precomputed")
    print 'Performing clustering...'
    labels = spectral.fit_predict(aff)

    # print out all labels
    for i in xrange(k):
        find_idx = np.where(labels==i)[0]
        print 'The list of classes in cluster {}'.format(i)
        print [imdb.classes[id] for id in find_idx]
        print '--------------------------------------------'

    return labels

if __name__ == '__main__':
    # TODO: debug code if necessary
       
    pass

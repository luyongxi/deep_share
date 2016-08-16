# Written by Yongxi Lu

""" Experiment and visualize clustering of class labels """

import numpy as np
from utils.blob import im_list_to_blob
from utils.timer import Timer
from utils.config import cfg
from sklearn.cluster import SpectralClustering

def Binary2Corr(label):
    """ Given an binary label matrix, convert it to correlation matrix.
        label: N by C matrix in {0,1}. 
        cm: C by C matrix in [-1,1].
    """
     # covert to {-1, 1} format
    label = 2.0*label-1.0
    cm = np.dot(label.transpose(), label)
    N = label.shape[0]
    cm = cm/N
    return cm

def MultiLabel_CM(net, imdb, postfix='', cls_idx=None, type='ecm'):
    """ Get Multi-label Label Correlation Matrix (LCM) """
    # class list is an ordered list of class index (onto the given dataset)
    if cls_idx is None:
        cls_idx = np.arange(imdb.num_classes)
    num_classes = len(cls_idx)
    num_images = imdb.num_images

    # iterate over images, collect error vectors
    label = np.zeros((num_images, num_classes)) # in {0,1} format
    timer = Timer()
    for i in xrange(num_images):
        # prepare blobs 
        label_name = "score"+postfix
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
        if type == 'ecm':
            label[i,:] = imdb.evaluate(scores, np.array([i]), cls_idx)
        elif type == 'lcm':
            label[i,:] = (scores>0.5).astype(np.float32, copy=False)
        # print infos
        print 'Image {}/{} ::: speed: {:.3f}s /iter'.format(i, num_images, timer.average_time)

    # get error correlation matrix
    print 'Computing {}...'.format(type)
    cm = Binary2Corr(label)
    
    return cm

def ClusterAffinity(aff, k, imdb, cls_idx):
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

# TODO: implement a function that loads the linear combinations of a model

# TODO: implement a function that computes similiarty based on this linear combination.

# Example codes
  # spectral = cluster.SpectralClustering(n_clusters=2,
  #                                         eigen_solver='arpack',
  #                                         affinity="nearest_neighbors")

if __name__ == '__main__':
    # TODO: first run some demo codes to ensure spectral clustering
    # itself is working. 
       
    pass

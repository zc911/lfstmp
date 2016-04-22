# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import sys
import os
sys.path.insert(0, os.getcwd() + '/python')
import caffe
import numpy as np
#import yaml
class CropLayer(caffe.Layer):
    def setup(self, bottom, top):
        #print 'setup'
        data = bottom[0].data
        size = bottom[1].data.shape
        assert(data.shape[0] == size[0])
        assert(data.shape[1] == size[1])
        assert(data.shape[2] >= size[2])
        assert(data.shape[3] >= size[3])

        #print 'size', size
        #sys.stdout.flush()
        top[0].reshape(size[0], size[1], size[2], size[3])


    def forward(self, bottom, top):
        print 'forward'
        data = bottom[0].data
        size = bottom[1].data.shape
        assert(data.shape[0] == size[0])
        assert(data.shape[1] == size[1])
        assert(data.shape[2] >= size[2])
        assert(data.shape[3] >= size[3])

        start_y = (data.shape[2] - size[2])/2
        end_y = start_y + size[2]

        start_x = (data.shape[3] - size[3])/2
        end_x = start_x + size[3]

        top[0].reshape(size[0], size[1], size[2], size[3])
        print 'top[0].shape', top[0].shape
        sys.stdout.flush()
        top[0].data[...] = data[:,:,start_y:end_y, start_x:end_x]
        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        if propagate_down[0]:
            print 'backward in python'
            sys.stdout.flush()
            data = bottom[0].data
            size = bottom[1].data.shape
            assert(data.shape[0] == size[0])
            assert(data.shape[1] == size[1])
            assert(data.shape[2] >= size[2])
            assert(data.shape[3] >= size[3])

            start_y = (data.shape[2] - size[2])/2
            end_y = start_y + size[2]

            start_x = (data.shape[3] - size[3])/2
            end_x = start_x + size[3]

            big_diff = np.zeros(data.shape)
            #diff = np.ones(size)
            print 'top diff size', top[0].diff.shape
            big_diff[:,:,start_y:end_y, start_x:end_x] = top[0].diff
            
            #print 'haha'
            bottom[0].diff[...] = big_diff

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


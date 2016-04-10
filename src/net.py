#!/usr/bin/env python
caffe_root = '/home/jiangliang/code/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import argparse
import matplotlib.pyplot as plt

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("network", help = 'network to test')
    parser.add_argument("-s", "--snapshot", 
            help = 'snapshot the network to use', 
            action = 'store', default = "", dest = "snapshot")
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    network = args.network
    snapshot = args.snapshot
    if snapshot == "":
        net = caffe.Net(network, caffe.TRAIN)
    else:
        net = caffe.Net(network, snapshot, caffe.TRAIN)
    print 'shape of data:'
    total_size = 0
    for k, v in net.blobs.items():
        size = 1
        for i in range(len(v.data.shape)):
            size = size * v.data.shape[i]
        print("{}, {}, ({})".format(k, v.data.shape, size))
        total_size = total_size + size
    print("total_size: {}".format(total_size))
    
    print 'shape of params: '
    param_num = 0
    for k, v in net.params.items():
        num = 1
        for i in range(len(v[0].data.shape)):
            num = num * v[0].data.shape[i]
        print("{}, {}, ({})".format(k, v[0].data.shape, num))
        param_num = param_num + num
    print "num of params: {}".format(param_num)
    param = net.params["scale_conv1_down"][0].data
    plt.figure(1)
    plt.hist(param, 50, facecolor = 'g')
    plt.title("param of scale_conv1_down")
    bias = net.params["scale_conv1_down"][1].data
    plt.figure(2)
    plt.hist(bias, 50, facecolor = 'g')
    plt.title("bias of scale_conv1_down")

    param = net.params["scale_conv2_down"][0].data
    plt.figure(3)
    plt.hist(param, 50, facecolor = 'g')
    plt.title("param of scale_conv2_down")
    bias = net.params["scale_conv2_down"][1].data
    plt.figure(4)
    plt.hist(bias, 50, facecolor = 'g')
    plt.title("bias of scale_conv2_down")

    param = net.params["scale_conv3_down"][0].data
    plt.figure(5)
    plt.hist(param, 50, facecolor = 'g')
    plt.title("param of scale_conv3_down")
    bias = net.params["scale_conv3_down"][1].data
    plt.figure(6)
    plt.hist(bias, 50, facecolor = 'g')
    plt.title("bias of scale_conv3_down")
    plt.show()

    print "finish"
if __name__ == "__main__":
    main()

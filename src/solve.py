#!/usr/bin/env python
import sys
import caffe
import argparse
import subprocess
import numpy as np

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_num",
            help = 'number of network used to solve',
            action = 'store', dest = 'net_num',
            default = 3, type = int)
    parser.add_argument("--net_prefix",
            help = 'path of networks',
            action = 'store', dest = 'net_prefix', 
            default = '''/home/liangjiang/code/\
residual_network/prototxt/DirectNet''')
    parser.add_argument("--solver_prefix",
            help = 'path of solver',
            action = 'store', dest = 'solver_prefix', 
            default = '/home/liangjiang/code/\
residual_network/prototxt/DirNet-cifar')
    return parser

def clean(net_num):
    print "cleaning"
    for i in range(net_num):
        child = subprocess.Popen("rm -rf /home/liangjiang/\
code/residual_network/results/loss/DirNet-cifar{}".format(i), shell = True)
        child.wait()
        child = subprocess.Popen("rm -rf /home/liangjiang/\
code/residual_network/results/snapshots/DirNet-cifar{}".format(i), shell = True)
        child.wait()

def init(net_num):
    for i in range(net_num):
        child = subprocess.Popen("mkdir -p /home/liangjiang/\
code/residual_network/results/\
loss/DirNet-cifar{}".format(i), shell = True)
        child = subprocess.Popen("mkdir -p /home/liangjiang/\
code/residual_network/results/\
snapshots/DirNet-cifar{}".format(i), shell = True)

def main():
    clean()
    max_iter = 20000
    test_iter = 200
    record_iter = 1
    threshold = -100
    prototxt_suffix = '.prototxt'
    parser = argparser()
    args = parser.parse_args()
    net_num = args.net_num
    net_prefix = args.net_prefix
    solver_prefix = args.solver_prefix
    cmd = "/home/liangjiang/code/residual_network\
/src/solve_network.py"
    snapshot_prefix = "/home/liangjiang/\
code/residual_network/\
results/snapshots/DirNet-cifar"
    loss_prefix = "/home/liangjiang/\
code/residual_network/results/\
loss/DirNet-cifar"
    print("cmd: {}".format(cmd))
    print("snapshot_prefix: {}".format(snapshot_prefix))
    print("loss_prefix: {}".format(loss_prefix))
    it = 0
    min_loss = 0
    init(net_num)
    caffe.set_mode_gpu()
    caffe.set_device(0)
    min_loss = 0
    for i in range(net_num):
        print i
        loss_path = "{}{}/DirNet-cifar-loss".format(loss_prefix, i)
        execute_cmd = ""
        if i == 0:
            execute_cmd = cmd \
            + " " + solver_prefix + str(i) + "-solver.prototxt" \
            + " --loss " + loss_path \
            + " --threshold " + str(threshold) \
            + " --max_iter " + str(max_iter) \
            + " --record_iter " + str(record_iter) \
            + " --test_iter " + str(test_iter) \
            + " -e"
        else:
            snapshot = snapshot_prefix \
                     + str(i - 1) + "/*.caffemodel"
            execute_cmd = cmd \
            + " " + solver_prefix + str(i) + "-solver.prototxt" \
            + " --loss " + loss_path \
            + " --snapshot " + snapshot \
            + " --threshold " + str(threshold) \
            + " --max_iter " + str(max_iter)\
            + " --record_iter " + str(record_iter) \
            + " --test_iter " + str(test_iter)\
            + " -e"\


        print execute_cmd
        try:
            child = subprocess.Popen(execute_cmd, shell = True)
            child.wait()
        except:
            print "Subprocess Error"
            sys.exit()
        preloss = np.loadtxt(loss_path)
        if i == 0:
            minloss = preloss[-1]
        else:
            if minloss > preloss[-1]:
               minloss = preloss[-1]
        print("minloss: {}".format(minloss))
if __name__ == '__main__':
    main()

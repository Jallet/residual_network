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
    child = subprocess.Popen("mkdir -p /home/liangjiang/\
code/residual_network/results/\
loss/DirNet-cifar", shell = True)
    child.wait()
    child = subprocess.Popen("mkdir -p /home/liangjiang/\
code/residual_network/results/\
accuracy/DirNet-cifar", shell = True)
    child.wait()

    for i in range(net_num):
        child = subprocess.Popen("mkdir -p /home/liangjiang/\
code/residual_network/results/\
loss/DirNet-cifar{}".format(i), shell = True)
        child.wait()
        child = subprocess.Popen("mkdir -p /home/liangjiang/\
code/residual_network/results/\
accuracy/DirNet-cifar{}".format(i), shell = True)
        child.wait()
        child = subprocess.Popen("mkdir -p /home/liangjiang/\
code/residual_network/results/\
snapshots/DirNet-cifar{}".format(i), shell = True)
        child.wait()

def main():
    max_iter = 6000
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
    acc_prefix = "/home/liangjiang/\
code/residual_network/results/\
accuracy/DirNet-cifar"
    print("cmd: {}".format(cmd))
    print("snapshot_prefix: {}".format(snapshot_prefix))
    print("loss_prefix: {}".format(loss_prefix))
    it = 0
    min_loss = 0
    clean(net_num)
    init(net_num)
    caffe.set_mode_gpu()
    caffe.set_device(0)
    min_loss = 0
    total_loss = []
    total_acc = []
    total_loss_path = "/home/liangjiang/\
code/residual_network/results/\
loss/DirNet-cifar/DirNet-cifar-loss"
    total_acc_path = "/home/liangjiang/\
code/residual_network/results/\
accuracy/DirNet-cifar/DirNet-cifar-acc"
    print("cmd: {}".format(cmd))
    for i in range(net_num):
        print i
        loss_path = "{}{}/DirNet-cifar-loss".format(loss_prefix, i)
        acc_path = "{}{}/DirNet-cifar-acc".format(acc_prefix, i)
        execute_cmd = ""
        if i == 0:
            execute_cmd = cmd \
            + " " + solver_prefix + str(i) + "-solver.prototxt" \
            + " --loss " + loss_path \
            + " --accuracy " + acc_path \
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
            + " --accuracy " + acc_path \
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
        preacc = np.loadtxt(acc_path)
        total_loss = np.hstack((total_loss, preloss))
        total_acc = np.hstack((total_acc, preacc))
        if i == 0:
            minloss = preloss[-1]
        else:
            if minloss > preloss[-1]:
               minloss = preloss[-1]
        print("minloss: {}".format(minloss))
        np.savetxt(total_loss_path, total_loss)
        np.savetxt(total_acc_path, total_acc)
if __name__ == '__main__':
    main()

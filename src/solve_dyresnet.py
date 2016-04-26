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
            default = 9, type = int)
    parser.add_argument("--net_prefix",
            help = 'path of networks',
            action = 'store', dest = 'net_prefix', 
            default = 'prototxt/DyResNet/ResNet-cifar')
    parser.add_argument("--solver_prefix",
            help = 'path of solver',
            action = 'store', dest = 'solver_prefix', 
            default = 'prototxt/DyResNet/ResNet-cifar')
    parser.add_argument("-m", "--max_iter",
            help = 'maximum iterations the network will train',
            action = 'append', dest = "max_iter")
    return parser

def clean(net_num):
    print "cleaning"
    child = subprocess.Popen("rm -rf results/loss/DyResNet-cifar", shell = True)
    child.wait()
    child = subprocess.Popen("rm -rf results/accuracy/DyResNet-cifar", shell = True)
    child.wait()
    child = subprocess.Popen("rm -rf results/snapshots/DyResNet-cifar", shell = True)
    child.wait()
    for i in range(net_num):
        child = subprocess.Popen("rm -rf results/loss/DyResNet-cifar{}".format(i), shell = True)
        child.wait()
        child = subprocess.Popen("rm -rf results/snapshots/DyResNet-cifar{}".format(i), shell = True)
        child.wait()
        child = subprocess.Popen("rm -rf results/snapshots/DyResNet-cifar{}".format(i), shell = True)
        child.wait()
        child = subprocess.Popen("rm -rf data/DyResNet-cifar{}/*".format(i+1), shell = True)
        child.wait()

def init(net_num):
    child = subprocess.Popen("mkdir -p results/loss/DyResNet-cifar", shell = True)
    child.wait()
    child = subprocess.Popen("mkdir -p results/accuracy/DyResNet-cifar", shell = True)
    child.wait()
    child = subprocess.Popen("mkdir -p results/snapshots/DyResNet-cifar", shell = True)
    child.wait()

    for i in range(net_num):
        child = subprocess.Popen("mkdir -p results/loss/DyResNet-cifar{}".format(i), shell = True)
        child.wait()
        child = subprocess.Popen("mkdir -p results/accuracy/DyResNet-cifar{}".format(i), shell = True)
        child.wait()
        child = subprocess.Popen("mkdir -p results/snapshots/DyResNet-cifar{}".format(i), shell = True)
        child.wait()

def main():
    default_max_iter = 6000
    batch_size = 250
    max_iters = [15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000]
#    max_epochs = [75, 75, 75 ,75, 75, 75, 75, 75, 75]
    output_layers = ["conv1", "conv2_1", "conv2_2", "conv2_3", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2"]
    test_iter = 200
    record_iter = 20
    threshold = -100
    prototxt_suffix = '.prototxt'
    parser = argparser()
    args = parser.parse_args()
    net_num = args.net_num
    net_prefix = args.net_prefix
    solver_prefix = args.solver_prefix
    cmd = "src/solve_network.py"
    snapshot_prefix = "results/snapshots/DyResNet-cifar"
    loss_prefix = "results/loss/DyResNet-cifar"
    acc_prefix = "results/accuracy/DyResNet-cifar"
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
    total_train_loss = []
    total_train_acc = []
    total_val_loss = []
    total_val_acc = []
    total_train_loss_path = "results/loss/DyResNet-cifar/DyResNet-cifar-loss.train"
    total_train_acc_path = "results/accuracy/DyResNet-cifar/DyResNet-cifar-acc.train"
    total_val_loss_path = "results/loss/DyResNet-cifar/DyResNet-cifar-loss.val"
    total_val_acc_path = "results/accuracy/DyResNet-cifar/DyResNet-cifar-acc.val"
    print("cmd: {}".format(cmd))
    for i in range(net_num):
        max_iter = max_iters[i]
        print max_iter
        print "Training DyResNet-cifar", i
        loss_path = "{}{}/DyResNet-cifar-loss".format(loss_prefix, i)
        train_output_path = "data/DyResNet-cifar{}/DyResNet_{}_train_lmdb".format(i + 1, output_layers[i])
        test_output_path = "data/DyResNet-cifar{}/DyResNet_{}_test_lmdb".format(i + 1, output_layers[i])
        train_loss = loss_path + ".train"
        val_loss = loss_path + ".val"
        acc_path = "{}{}/DyResNet-cifar-acc".format(acc_prefix, i)
        train_acc = acc_path + ".train"
        val_acc = acc_path + ".val"
        execute_cmd = ""
        if i == 0:
            execute_cmd = cmd \
            + " " + solver_prefix + str(i) + "-solver.prototxt" \
            + " " + train_output_path \
            + " " + test_output_path \
            + " --train-loss " + train_loss \
            + " --val-loss " + val_loss \
            + " --train-acc " + train_acc \
            + " --val-acc " + val_acc \
            + " --threshold " + str(threshold) \
            + " --max_iter " + str(max_iter) \
            + " --record_iter " + str(record_iter) \
            + " --test_iter " + str(test_iter) \
            + " --batch_size " + str(batch_size) \
            + " --output_layer " + output_layers[i] 
        else:
            snapshot = snapshot_prefix \
                     + str(i - 1) + "/*.caffemodel"
            execute_cmd = cmd \
            + " " + solver_prefix + str(i) + "-solver.prototxt" \
            + " " + train_output_path \
            + " " + test_output_path \
            + " --train-loss " + train_loss \
            + " --val-loss " + val_loss \
            + " --train-acc " + train_acc \
            + " --val-acc " + val_acc \
            + " --snapshot " + snapshot \
            + " --threshold " + str(threshold) \
            + " --max_iter " + str(max_iter)\
            + " --record_iter " + str(record_iter) \
            + " --test_iter " + str(test_iter)\
            + " --batch_size " + str(batch_size) \
            + " --output_layer " + output_layers[i] 


        print execute_cmd
        try:
            child = subprocess.Popen(execute_cmd, shell = True)
            child.wait()
        except:
            print "Subprocess Error"
            sys.exit()

        pre_train_loss = np.loadtxt(train_loss)
        pre_train_acc = np.loadtxt(train_acc)
        total_train_loss = np.hstack((total_train_loss, pre_train_loss))
        total_train_acc = np.hstack((total_train_acc, pre_train_acc))
        pre_val_loss = np.loadtxt(val_loss)
        pre_val_acc = np.loadtxt(val_acc)
        total_val_loss = np.hstack((total_val_loss, pre_val_loss))
        total_val_acc = np.hstack((total_val_acc, pre_val_acc))

        if i == 0:
            minloss = pre_train_loss[-1]
        else:
            if minloss > pre_train_loss[-1]:
               minloss = pre_train_loss[-1]
        print("minloss: {}".format(minloss))
        np.savetxt(total_train_loss_path, total_train_loss)
        np.savetxt(total_train_acc_path, total_train_acc)
        np.savetxt(total_val_loss_path, total_val_loss)
        np.savetxt(total_val_acc_path, total_val_acc)
if __name__ == '__main__':
    main()

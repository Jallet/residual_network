#!/usr/bin/env python
import sys
import caffe
import argparse
import numpy as np

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("solver", 
            help = "Path to the solver to use")
    parser.add_argument("--train-loss", "-tl", 
            help = "Path to store trainning loss",
            action = "store", dest = "train_loss",
            default = "results/train-loss")
    parser.add_argument("--train-accuracy", "-ta", 
            help = "Path to store accuracy",
            action = "store", dest = "train_accuracy",
            default = "results/train-accuracy")
    parser.add_argument("--val-loss", "-vl", 
            help = "Path to store valning loss",
            action = "store", dest = "val_loss",
            default = "results/val-loss")
    parser.add_argument("--val-accuracy", "-va", 
            help = "Path to store accuracy",
            action = "store", dest = "val_accuracy",
            default = "results/val-accuracy")
    parser.add_argument("--snapshot", "-s", 
            help = "Path to the snapshot to resume",
            action = "store", dest = "snapshot",
            default = "")
    parser.add_argument("--threshold",
            help = "Threshold to stop trainning early",
            action = "store", dest = "threshold", 
            type = int, default = 1e-3)
    parser.add_argument("--max_iter", "-m", 
            help = "maximium training iterations",
            action = "store", dest = "max_iter",
            type = int, default = 60000)
    parser.add_argument("--test_iter", 
            help = "iterations to test early stop",
            action = "store", dest = "test_iter",
            type = int, default = 200)
    parser.add_argument("--record_iter", "-r",  
            help = "iterations to record loss",
            action = "store", dest = "record_iter",
            type = int, default = 20)
    parser.add_argument("--early-stop", "-e", 
            help = "Whether stop when loss does not decrease",
            action = "store_true", dest = "early_stop")
    return parser

def main():
    #read arguments
    parser = argparser()
    args = parser.parse_args()
    solver = args.solver
    train_loss_path = args.train_loss
    train_acc_path = args.train_accuracy
    val_loss_path = args.val_loss
    val_acc_path = args.val_accuracy
    snapshot = args.snapshot
    threshold = args.threshold
    max_iter = args.max_iter
    record_iter = args.record_iter
    test_iter = args.test_iter
    early_stop = args.early_stop 

    caffe.set_mode_gpu()
    caffe.set_device(0)
    
    solver = caffe.get_solver(solver)

    if snapshot != "":
        solver.net.copy_from(snapshot)
    train_loss = np.zeros(max_iter / record_iter) 
    train_acc = np.zeros(max_iter / record_iter) 
    val_loss = np.zeros(max_iter / record_iter) 
    val_acc = np.zeros(max_iter / record_iter) 
    pre_loss = 0
    pre_acc = 0

    cur_train_loss = np.zeros(record_iter)
    cur_train_acc = np.zeros(record_iter)
    cur_val_loss = np.zeros(record_iter)
    cur_val_acc = np.zeros(record_iter)

    for i in range(max_iter):
        solver.step(1)
        cur_train_loss[i % record_iter] = solver.net.blobs["loss"].data
        cur_train_acc[i % record_iter] = solver.net.blobs["accuracy"].data
        cur_val_loss[i % record_iter] = solver.test_nets[0].blobs["loss"].data
        cur_val_acc[i % record_iter] = solver.test_nets[0].blobs["accuracy"].data
        if i % record_iter == record_iter - 1:
            train_loss[i / record_iter] = np.average(cur_train_loss)
            train_acc[i / record_iter] = np.average(cur_train_acc)
            val_loss[i / record_iter] = np.average(cur_val_loss)
            val_acc[i / record_iter] = np.average(cur_val_acc)

            cur_train_loss = np.zeros(record_iter)
            cur_train_acc = np.zeros(record_iter)
            cur_val_loss = np.zeros(record_iter)
            cur_val_acc = np.zeros(record_iter)

            train_temp_loss = train_loss[0 : i / record_iter + 1]
            train_temp_acc = train_acc[0 : i / record_iter + 1]
            val_temp_loss = val_loss[0 : i / record_iter + 1]
            val_temp_acc = val_acc[0 : i / record_iter + 1]
            np.savetxt(train_loss_path, train_temp_loss)
            np.savetxt(train_acc_path, train_temp_acc)
            np.savetxt(val_loss_path, val_temp_loss)
            np.savetxt(val_acc_path, val_temp_acc)
        #if early_stop and (i % test_iter == 0):
        #    if i == 0:
        #        pre_train_loss = cur_train_loss
        #        pre_train_acc = cur_train_acc
        #        pre_val_loss = cur_val_loss
        #        pre_val_acc = cur_val_acc
        #    elif (pre_train_loss - cur_train_loss) / pre_train_loss < threshold:
        #        print "Converged, stopping...."
    solver.snapshot()

if __name__ == "__main__":
    main()

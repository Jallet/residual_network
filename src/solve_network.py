#!/usr/bin/env python
import sys
sys.path.insert(0, '/usr/local/caffe/python/')
import caffe
import argparse
import numpy as np

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("solver", 
            help = "Path to the solver to use")
    parser.add_argument("--loss", 
            help = "Path to store loss",
            action = "store", dest = "loss",
            default = "/home/liangjiang/code/residual_network\
                    /results/loss")
    parser.add_argument("--snapshot",
            help = "Path to the snapshot to resume",
            action = "store", dest = "snapshot",
            default = "")
    parser.add_argument("--threshold",
            help = "Threshold to stop trainning early",
            action = "store", dest = "threshold", 
            type = int, default = 1e-3)
    parser.add_argument("--max_iter", 
            help = "maximium training iterations",
            action = "store", dest = "max_iter",
            type = int, default = 60000)
    parser.add_argument("--test_iter", 
            help = "iterations to test early stop",
            action = "store", dest = "test_iter",
            type = int, default = 200)
    parser.add_argument("--record_iter", 
            help = "iterations to record loss",
            action = "store", dest = "record_iter",
            type = int, default = 20)
    return parser

def main():
    #read arguments
    parser = argparser()
    args = parser.parse_args()
    solver = args.solver
    loss_path = args.loss
    snapshot = args.snapshot
    threshold = args.threshold
    max_iter = args.max_iter
    record_iter = args.record_iter
    test_iter = args.test_iter
    
    solver = caffe.get_solver(solver)
    if snapshot != "":
        solver.net.copy_from(snapshot)
    loss = np.zeros(max_iter / record_iter) 
    pre_loss = 0
    for i in range(max_iter):
        solver.step(1)
        cur_loss = solver.net.blobs["loss"].data
        if i % record_iter == 0:
            loss[i / record_iter] = cur_loss
        if i % test_iter == 0:
            if i == 0:
                pre_loss = cur_loss
            elif (pre_loss - cur_loss) / pre_loss < threshold:
                print "Converged, stopping...."
    solver.snapshot()
    loss = loss[0 : i / record_iter]
    np.savetxt(loss_path, loss)

if __name__ == "__main__":
    main()

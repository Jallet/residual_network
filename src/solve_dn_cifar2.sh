#!/usr/bin/env sh
#clean
#rm -r /home/liangjiang/code/residual_network/results/loss/DirNet-cifar2
rm -r /home/liangjiang/code/residual_network/results/snapshots/DirNet-cifar2
mkdir -p /home/liangjiang/code/residual_network/results/loss/DirNet-cifar2
mkdir -p  /home/liangjiang/code/residual_network/results/accuracy/DirNet-cifar2
mkdir -p  /home/liangjiang/code/residual_network/results/snapshots/DirNet-cifar2

/home/liangjiang/code/residual_network/src/solve_network.py /home/liangjiang/code/residual_network/prototxt/DirNet-cifar2-solver.prototxt --train-loss /home/liangjiang/code/residual_network/results/loss/DirNet-cifar2/DirNet-cifar-train-loss --train-acc /home/liangjiang/code/residual_network/results/accuracy/DirNet-cifar2/DirNet-cifar-train-acc --val-loss /home/liangjiang/code/residual_network/results/loss/DirNet-cifar2/DirNet-cifar-val-loss --val-acc /home/liangjiang/code/residual_network/results/accuracy/DirNet-cifar2/DirNet-cifar-val-acc --max_iter 20000 

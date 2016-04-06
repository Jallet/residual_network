#!/usr/bin/env sh
#clean
rm -r /home/liangjiang/code/residual_network/results/loss/DirNet-cifar2
rm -r /home/liangjiang/code/residual_network/results/snapshots/DirNet-cifar2
mkdir -p /home/liangjiang/code/residual_network/results/loss/DirNet-cifar2
mkdir -p  /home/liangjiang/code/residual_network/results/snapshots/DirNet-cifar2

/home/liangjiang/code/residual_network/src/solve_network.py /home/liangjiang/code/residual_network/prototxt/DirNet-cifar2-solver.prototxt -l /home/liangjiang/code/residual_network/results/loss/DirNet-cifar2/loss --max_iter 20000 

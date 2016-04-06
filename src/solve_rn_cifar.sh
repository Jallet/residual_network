#!/usr/bin/env sh
#clean
rm -r /home/liangjiang/code/residual_network/results/loss/ResNet-cifar
rm -r /home/liangjiang/code/residual_network/results/snapshots/ResNet-cifar
mkdir -p /home/liangjiang/code/residual_network/results/loss/ResNet-cifar
mkdir -p  /home/liangjiang/code/residual_network/results/snapshots/ResNet-cifar

/home/liangjiang/code/residual_network/src/solve_network.py /home/liangjiang/code/residual_network/prototxt/ResNet-cifar-solver.prototxt -l /home/liangjiang/code/residual_network/results/loss/ResNet-cifar/loss --max_iter 20000 

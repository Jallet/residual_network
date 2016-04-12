#!/usr/bin/env sh
#clean
#rm -r /home/liangjiang/code/residual_network/results/loss/ResNet-cifar
rm -r /home/liangjiang/code/residual_network/results/snapshots/ResNet-cifar
mkdir -p /home/liangjiang/code/residual_network/results/loss/ResNet-cifar
mkdir -p /home/liangjiang/code/residual_network/results/accuracy/ResNet-cifar
mkdir -p  /home/liangjiang/code/residual_network/results/snapshots/ResNet-cifar

/home/liangjiang/code/residual_network/src/solve_network.py /home/liangjiang/code/residual_network/prototxt/ResNet-cifar-solver.prototxt --train-loss /home/liangjiang/code/residual_network/results/loss/ResNet-cifar/ResNet-cifar-train-loss --train-acc /home/liangjiang/code/residual_network/results/accuracy/ResNet-cifar/ResNet-cifar-train-acc --val-loss /home/liangjiang/code/residual_network/results/loss/ResNet-cifar/ResNet-cifar-val-loss --val-acc /home/liangjiang/code/residual_network/results/accuracy/ResNet-cifar/ResNet-cifar-val-acc --max_iter 60000

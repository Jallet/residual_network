#!/usr/bin/env sh
#clean
#rm -r /home/liangjiang/code/residual_network/results/loss/ResNet-cifar
rm -r /home/liangjiang/code/residual_network/results/snapshots/PlainNet-cifar
mkdir -p /home/liangjiang/code/residual_network/results/loss/PlainNet-cifar
mkdir -p /home/liangjiang/code/residual_network/results/accuracy/PlainNet-cifar
mkdir -p  /home/liangjiang/code/residual_network/results/snapshots/PlainNet-cifar

/home/liangjiang/code/residual_network/src/solve_network.py /home/liangjiang/code/residual_network/prototxt/PlainNet-cifar-solver.prototxt --train-loss /home/liangjiang/code/residual_network/results/loss/PlainNet-cifar/PlainNet-cifar-train-loss --train-acc /home/liangjiang/code/residual_network/results/accuracy/PlainNet-cifar/PlainNet-cifar-train-acc --val-loss /home/liangjiang/code/residual_network/results/loss/PlainNet-cifar/PlainNet-cifar-val-loss --val-acc /home/liangjiang/code/residual_network/results/accuracy/PlainNet-cifar/PlainNet-cifar-val-acc --max_iter 20000 

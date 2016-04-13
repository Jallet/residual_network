#!/usr/bin/env sh
#clean
#rm -r results/loss/ResNet-cifar8
rm -r results/snapshots/ResNet-cifar8
mkdir -p results/loss/ResNet-cifar8
mkdir -p results/accuracy/ResNet-cifar8
mkdir -p results/snapshots/ResNet-cifar8

src/solve_network.py prototxt/DyResNet/ResNet-cifar8-solver.prototxt --train-loss results/loss/ResNet-cifar8/ResNet-cifar8-train-loss --train-acc results/accuracy/ResNet-cifar8/ResNet-cifar8-train-acc --val-loss results/loss/ResNet-cifar8/ResNet-cifar8-val-loss --val-acc results/accuracy/ResNet-cifar8/ResNet-cifar8-val-acc --max_iter 900

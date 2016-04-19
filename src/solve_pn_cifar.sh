#!/usr/bin/env sh
#clean
#rm -r results/loss/ResNet-cifar
rm -r results/snapshots/PlainNet-cifar
mkdir -p results/loss/PlainNet-cifar
mkdir -p results/accuracy/PlainNet-cifar
mkdir -p  results/snapshots/PlainNet-cifar

src/solve_network.py prototxt/PlainNet-cifar-solver.prototxt --train-loss results/loss/PlainNet-cifar/PlainNet-cifar-train-loss --train-acc results/accuracy/PlainNet-cifar/PlainNet-cifar-train-acc --val-loss results/loss/PlainNet-cifar/PlainNet-cifar-val-loss --val-acc results/accuracy/PlainNet-cifar/PlainNet-cifar-val-acc --max_iter 60000 

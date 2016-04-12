#!/usr/bin/env sh
#clean
#rm -r results/loss/DirNet-cifar2
rm -r results/snapshots/DirNet-cifar2
mkdir -p results/loss/DirNet-cifar2
mkdir -p  results/accuracy/DirNet-cifar2
mkdir -p  results/snapshots/DirNet-cifar2

src/solve_network.py prototxt/DirNet-cifar2-solver.prototxt --train-loss results/loss/DirNet-cifar2/DirNet-cifar-train-loss --train-acc results/accuracy/DirNet-cifar2/DirNet-cifar-train-acc --val-loss results/loss/DirNet-cifar2/DirNet-cifar-val-loss --val-acc results/accuracy/DirNet-cifar2/DirNet-cifar-val-acc --max_iter 20000 

#!/usr/bin/env sh
#clean
#rm -r results/loss/ResNet-cifar
rm -r results/snapshots/ResNet-cifar
mkdir -p results/loss/ResNet-cifar
mkdir -p results/accuracy/ResNet-cifar
mkdir -p results/snapshots/ResNet-cifar

src/solve_network.py prototxt/ResNet-cifar-solver.prototxt --train-loss results/loss/ResNet-cifar/ResNet-cifar-train-loss --train-acc results/accuracy/ResNet-cifar/ResNet-cifar-train-acc --val-loss results/loss/ResNet-cifar/ResNet-cifar-val-loss --val-acc results/accuracy/ResNet-cifar/ResNet-cifar-val-acc --output_layer conv1 --batch_size 200 --max_iter 250 --epoch 80 data/ResNet-cifar/ResNet-cifar_train_lmdb data/ResNet-cifar/ResNet-cifar_test_lmdb

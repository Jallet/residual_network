#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    return parser
parser = argparser()
args = parser.parse_args()
path = args.path

data = np.loadtxt(path)
plt.figure(1)
plt.plot(data)
plt.legend()
plt.title("data")
plt.show()

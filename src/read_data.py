#!/usr/bin/env python
import sys
caffe_root = '/home/liangjiang/code/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb
import argparse
from caffe.proto import caffe_pb2

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("lmdb_file", help = "lmdb file to use")
    return parser

def main():
    parser = argparser()
    args = parser.parse_args() 
    lmdb_file = args.lmdb_file
    data_lmdb = lmdb.open(lmdb_file, map_size = int(1e12))
    txn = data_lmdb.begin()
    cursor = txn.cursor()
    datum = caffe_pb2.Datum()

    print cursor.count()
    count = 0
    for key, value in cursor:
        count = count + 1
    print count

if __name__ == "__main__":
    main()

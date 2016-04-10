#!/usr/bin/env python
import argparse
import subprocess

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help = "log to use")
    parser.add_argument("output", help = "file to store result")
    parser.add_argument("-p", "--phase",
            help =  "which phase to get data, train or test\
                    0 stands for train, 1 stands for test",
            action = "store", default = 0, type = int) 
    parser.add_argument("-t", "--type", help = "which type of data to get, \
            0 stands for loss, 1 stands for accuracy",
            action = "store", default = 0, type = int)
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    log = args.log
    output = args.output
    phase = args.phase
    data_type = args.type
    target_str = "" 
    if phase == 0:
        if data_type == 0:
            print "Getting training loss"
            target_str = "Train net output #1"
        elif data_type == 1:
            target_str = "Train net output #0"
            print "Getting training accuracy"
    elif phase == 1:
        if data_type == 0:
            print "Getting testing loss"
            target_str = "Test net output #1"
        elif data_type == 1:
            print "Getting testing accuracy"
            target_str = "Test net output #0"

    cmd = "cat " + log + " | grep \'" + target_str \
    + "\' | awk -F \' \' \'{print $11}\' > " + output
    print cmd
    child = subprocess.Popen(cmd, shell = True) 
    child.wait()
    
if __name__ == "__main__":
    main()

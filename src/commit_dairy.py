#!/usr/bin/env python
import argparse
import subprocess
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', help = 'log to commit')
    parser.add_argument('--work', action = 'store', dest = 'work',
                        default = 'did not work', 
                        help = 'how well the model worked')
    parser.add_argument('--loss', action = 'store', dest = 'loss',
                        default = '-1',
                        help = 'loss of the model')
    return parser
def main():
    parser = argparser()
    args = parser.parse_args()
    work = args.work
    log = args.log
    loss = args.loss
    try:
        child = subprocess.Popen('git commit -a -m \"[DAIRY][' + work 
                                + '][loss = ' + loss + ']' + log + '\"', 
                                shell = True)
        child.wait()
        print "Commit succeeded"
    except subprocess.CalledProcessError:
        print "Commit failed"

if __name__ == '__main__':
    main()

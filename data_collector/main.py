#!/usr/bin/env python

import sys, subprocess, logging

def start_processes():
    subprocess.Popen([sys.executable, 'prices/price_collector.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

if __name__=='__main__':
    print('Starting processes..')
    start_processes()

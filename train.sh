#!/bin/sh
nohup python -u webshell.py >dctf.out 2>&1 & tail -f dctf.out

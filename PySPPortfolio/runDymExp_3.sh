#!/bin/bash
ns=$1
for alpha in 0.5 0.55 0.6 0.65 0.7
do 
    for p in 110
    do
       python dynamicSymbolSPPortfolio.py -n $ns -p $p -a $alpha  -f
    done
done

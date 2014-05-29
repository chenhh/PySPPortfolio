#!/bin/bash
for alpha in 0.9 0.95 0.99
do
    for p in 50
    do
        python fixedSymbolSPPortfolio.py -n 50 -p $p -a $alpha  -m 200501 -f
    done
done


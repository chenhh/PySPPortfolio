#!/bin/bash
for alpha in 0.5 0.55 0.6 0.65
do
    for p in 50
    do
        python fixedSymbolSPPortfolio.py -n 50 -p $p -a $alpha  -m 200501 -f
    done
done


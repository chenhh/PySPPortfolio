#!/bin/bash
for alpha in 0.7 0.75 0.8 0.85
do
    for p in 50
    do
        python fixedSymbolSPPortfolio.py -n 50 -p $p -a $alpha  -m 200501 -f
    done
done


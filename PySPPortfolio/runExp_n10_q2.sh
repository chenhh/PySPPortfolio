#!/bin/bash
for alpha in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99
do
    for p in 60 70 80 90
    do
        python fixedSymbolSPPortfolio.py -n 10 -p $p -a $alpha  -m 200501 -f
    done
done

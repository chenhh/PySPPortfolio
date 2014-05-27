#!/bin/bash
for alpha in 0.7 0.8 0.85 0.9 0.95 0.99
do
    for p in 90 
    do
        python fixedSymbolSPPortfolio.py -n 5 -p $p -a $alpha  -m 200501 -f
    done
done

for alpha in 0.65 0.7 0.8 0.85 0.9 0.95 0.99
do
    for p in 100 110 120
    do
        python fixedSymbolSPPortfolio.py -n 5 -p $p -a $alpha  -m 200501 -f
    done
done
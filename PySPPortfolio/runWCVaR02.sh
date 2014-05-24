#!/bin/bash
ns=$1
for alpha in 0.75 0.8 0.85 0.9 0.95
do
    python fixedSymbolSPPortfolio.py -n $ns -r WCVaR -s 100 -a $alpha  -m 200501 -f    
done
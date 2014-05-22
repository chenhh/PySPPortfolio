#!/bin/bash
ns=$1
for alpha in 0.5 0.55 0.6 0.65 0.7 
do
    python fixedSymbolSPPortfolio.py -n $ns -r WCVaR -s 100 -a $alpha  -m 200501 -f    
done
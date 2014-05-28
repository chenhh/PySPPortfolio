#!/bin/bash
for alpha in 0.8 0.85 0.9 0.95 0.99
do
    python fixedSymbolSPPortfolio.py -n 15 -p 50 -a $alpha  -m 200501 -f
 
done

for alpha in 0.85 0.9 0.95 0.99
do
    python fixedSymbolSPPortfolio.py -n 15 -p 60 -a $alpha  -m 200501 -f
 
done

 python fixedSymbolSPPortfolio.py -n 15 -p 120 -a 0.75  -m 200501 -f
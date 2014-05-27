#!/bin/bash
for alpha 0.95 0.99
do
    for p in 70 80
    do
        python fixedSymbolSPPortfolio.py -n 50 -p $p -a $alpha  -m 200501 -f
    done
done

python fixedSymbolSPPortfolio.py -n 50 -p 90 -a 0.99  -m 200501 -f
python fixedSymbolSPPortfolio.py -n 50 -p 100 -a 0.99  -m 200501 -f
python fixedSymbolSPPortfolio.py -n 50 -p 110 -a 0.99  -m 200501 -f
python fixedSymbolSPPortfolio.py -n 50 -p 120 -a 0.99  -m 200501 -f
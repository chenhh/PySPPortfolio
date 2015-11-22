required package
--------------------
  - numpy
  - scipy
  - pandas
  - Pyomo
  - simplejson
  - arch, "conda install -c https://conda.binstar.org/bashtage arch"
  
  
    gen unbiased scenarios:
        - n_stock: {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
        - win_length: {50, 60, ..., 240}
        - n_scenario:  200
        - biased: {unbiased,}
        - cnt: {1,2,3}
        - combinations: 10 * 20 * 3 = 600 (only unbiased)
    
       using three computers: 
        - comstar (linux mint 17.2, i7-2600, 16g), 
        - par56 (Ubuntu 14.04.3, i7-3770, 8g),
        - par67 (linux mint 17.2, i7-3770, 8g), 
       
       it requires about 30 hours
       
    solve unbiased min_cvar_sp:
       the same scenarios as above, and additional parameters:
       alphas = ('0.50', '0.55', '0.60', '0.65',
                  '0.70', '0.75', '0.80', '0.85',
                  '0.90', '0.95')         
       - combinations: 10 * 20 * 3 * 10 = 6000 (only unbiased)
       
       four computers:
            - comstar, par56, par67, and par48
            - par48 (gentoo linux, i5-2400, 16g)
        
       it requires about 70 hours
            
    
    
       
       
    
    
'''
Created on 2014/3/25

@author: chenhh
'''
from __future__ import division
from libc.math cimport (min, max, fabs)

cdef cubicSolve(double* xk):
    '''
    cubParam, np.array, size: 4
    '''
    cdef: 
        int IncrDevStep = 50    #how often we increase the START_DEV
        
        int i=0
        int j
        int nb_iter=0
        int indx[4]
        double gk[4]        #gradient
        double hessk[4][4]  #hessian
        double invhk[4][4]  #inverse hessian
        double ni           #infinite norm for the gradient
        double error1       #L2-error on the target moments, at initial pt
        double error2       #L2-error on the target moments, at final pt
        double x_ini[4]
        
    #Initialization of the starting point
    for j in xrange(4):
        cubParam[j]=x_ini[j]



cdef void momF1(double value[7], double *xk):
    cdef:
        int i, k
        double c

    for in xrange(7):
        value[i] = 0
    
    for k in xrange(4):
        c=xk[k]
        for i in xrange(7):
            value[i]+=c*InMom[i+k];
        

cdef void momF2(double value[7],double *xk):
    cdef:
        int i
        int k
        int l
        int ml1
        int ml2
        double c

    for i in xrange(7):
        value[i]=0
    
    for k in xrange(7):
            c=0
            ml1=min(3, k)
            ml2=max(0, k-3)
            for l in xrange(ml2, ml1+1):
                c+=xk[l]*xk[k-l]
            
            for i in xrange(7):
                value[i]+=c*InMom[k+i];


cdef void momF3(double value[7],double *xk):
    cdef:
        int i,k,l,m,ml1,ml2,mm1,mm2
        double c


    for i in xrange(4):
        value[i]=0
    
    
    for k in xrange(10):
        c=0
        ml1=min(3,k)
        ml2=max(0,k-6)
        for l in xrange(ml2, ml1+1):
            mm1=min(k-l,3);
            mm2=max(k-l-3,0)
            for m in xrange(mm2, mm1+1):   
                c+=xk[k-l-m]*xk[l]*xk[m]
        for i in xrange(4):
            value[i]+=c*InMom[k+i]
        

cdef double momF4(double *xk):
    cdef:
        double value = 0
        int k,l,m,n,ml1,ml2,mm1,mm2,mn1,mn2
        double c


    
    for k in xrange(13):
        c=0
        ml1=min(3,k)
        ml2=max(0,k-9)
        for l in xrange(ml2, ml1+1):
            mm1=min(k-l,3)
            mm2=max(k-l-6,0)
            for m in xrange(mm2, mm1+1):
                mn1=min(k-l-m,3)
                mn2=max(k-l-m-3,0)
                for n in xrange(mn2, mn1+1):
                    c+=xk[k-l-m-n]*xk[l]*xk[m]*xk[n];
        value+=c*InMom[k]
    
    return value

cdef double objval( double *xk):
    '''Function objval(x) computes and returns obj(x).'''
    cdef:
        double value = 0
        int i

    for i in xrange(4):
        value += pow(moment[i+1][0]-TgMom[i],2)
    
    return value


cdef void gradhessian(double c[4],double Q[4][4],double *xk):
    cdef:
        int i, j, k

    #first row initialisation
    for i in xrange(7):
        moment[0][i]=InMom[i]
    
    #initilisation of the other rows
    momF1(moment[1],xk)
    momF2(moment[2],xk)
    momF3(moment[3],xk)
    moment[4][0]=momF4(xk)

    #gradient initialisation
    for i in xrange(4):
        c[i]=0
    
    #value
    for i in xrange(4):
        for j in xrange(4):
            c[i]+=2*(j+1)*moment[j][i]*(moment[j+1][0]-TgMom[j])
  

    #hessian - no initialization needed, done for k=0
    for i in xrange(4):
        for j in xrange(i):
            Q[i][j] = 2*(moment[0][i]*moment[0][j])
            for k in xrage(1, 4):
                Q[i][j] += 2*(k+1)*((k+1)*moment[k][i]*moment[k][j]
                                    +k*moment[k-1][i+j]*(moment[k+1][0]-TgMom[k]))
            if j < i:
                Q[j][i]=Q[i][j];
 
 
cdef int migs (double a[4][4],double x[4][4],int indx[4]):
    ''' Function to invert matrix a[][] with the inverse stored
       in x[][] in the output.  copyright (c) Tao Pang 2001. 
    '''
    cdef:
        int i,j,k
        double b[N][N]

    for i in xrange(4):
        for j in xrnage(4):
            b[i][j] = 0
    
    for i in xrange(4):
        b[i][i] = 1
   
    if eigs(a, indx) > 0:
        return 1

    for i in xrange(3):
        for j in xrange(i+1, 4):
            for k in xrange(4):
                b[indx[j]][k] = b[indx[j]][k]-a[indx[j]][i]*b[indx[i]][k]
 
    for i in xrange(4):
        x[N-1][i] = b[indx[N-1]][i]/a[indx[N-1]][N-1]
        for j in reverse(xrange(0, N-1)):
            x[j][i] = b[indx[j]][i]
            for k in xrange(j+1, 4):
                x[j][i] = x[j][i]-a[indx[j]][k]*x[k][i]
            x[j][i] = x[j][i]/a[indx[j]][j]
  
    return 0


cdef int elgs (double a[4][4],int indx[4]):
    '''
    Function to perform the partial-pivoting Gaussian elimination.
    a[][] is the original matrix in the input and transformed
    matrix plus the pivoting element ratios below the diagonal
    in the output.  indx[] records the pivoting order.
    copyright (c) Tao Pang 2001.
    '''
    cdef:
        int i, j, k=4, itmp
        double c1, pi, pi1, pj
        double c[4]


    for i in xrange(4):
        indx[i] = i
 
    #Find the rescaling factors, one from each row 
    for i in xrange(4):
        c1 = 0
        for j in xrange(4):
            if fabs(a[i][j]) > c1:
                c1 = fabs(a[i][j])
        c[i] = c1

    #Search the pivoting (largest) element from each column 
    for j in xrange(3):
        pi1 = 0
        for i in xrange(j, N):
            pi = fabs(a[indx[i]][j])/c[indx[i]]
            if pi > pi1:
                pi1 = pi
                k = i

    # ADDED BY MK
    if k>=4: 
        return 1


    #Interchange the rows via indx[] to record pivoting order
    itmp = indx[j]
    indx[j] = indx[k]
    indx[k] = itmp
    for i in xrange(j+1, 4):
        pj = a[indx[i]][j]/a[indx[j]][j]

        #Record pivoting ratios below the diagonal */
        a[indx[i]][j] = pj

        #Modify other elements accordingly 
        for k in xrange(j+1, 4):
            a[indx[i]][k] = a[indx[i]][k]-pj*a[indx[j]][k]
      
    return 0


cdef void nextstep(double invhk[4][4],double gk[4],double xk[4])
    '''
    Function that do a multiplication matrix-vector, give the opposite vector of the result
    and add to the initial xk this direction vector. Used at the beginning of each iteration of
    the newton algorithm 
    '''
    cdef:
        int i,j
        double di=0

    for i in xrange(4):
        for j in xrange(4):
            di+=-invhk[i][j]*gk[j]
        xk[i]+=di
        di=0
  
cdef double norminf(double vecteur[4]):
    '''norminf computes the infinite norm of an array'''
    cdef:
        double val=0
        int i
        
    for i in xrange(4):
        if vecteur[i]>val:
            val=vecteur[i]
        elif -vecteur[i]>val:
            val=-vecteur[i]
    return val    
        

cdef int spofa(double mat[4][4]){
    ''' 
    spofa tests if the matrix is positive definite
    returns 1 if the matrix is positive definite and 0 otherwise
    '''
    cdef:
        int i,j,k
        double sum
        double p[N]
    
    for i in xrange(4):
        for j in xrange(i, 4):
            for k in reverse(xrange(0, i):
                s = mat[i][j]
                s-= mat[i][k]*mat[j][k]; 
        if i == j:
            if s <=0.0:
                return 0
            else:
                p[i] = sqrt(s)
        else:
            mat[j][i] = s/p[i]

    #Cholesky run to the end -> matrix is PD
    return 1
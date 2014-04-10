cdef extern from "matrix.h":
    
    ctypedef struct TVector:
        int size
        double *val

    ctypedef struct  TMatrix:
        int nrow, ncol
        double ** val

    #function
    cdef int Mat_Init(TMatrix* p_mat, int mb_rows, int nmb_cols)
    cdef int Vec_Init(TVector* p_vec, int size) 
    cdef void Mat_Display(TMatrix* p_A, char name[])
    cdef void Vec_Display(TVector* p_A, char name[])
    cdef void Vec_Kill(TVector* p_A)
    cdef void Mat_Kill(TMatrix* p_A)
    
    
cdef extern from "HKW_sg.h":
    int HKW_ScenGen(int FormatOfMoms,
                    TMatrix* p_TarMoms, #size: 4 * n_rv
                    TMatrix*  p_TgCorrs, #size: n_rv * n_rv
                    TVector*  p_Probs,
                    TMatrix*  p_OutMat,
                    double  MaxErrMom, double  MaxErrCorr,
                    int  TestLevel, int  MaxTrial,
                    int  MaxIter, int  UseStartValues,
                    double * p_errMom, double * p_errCorr,
                    int * p_nmbTrial, int * p_nmbIter)

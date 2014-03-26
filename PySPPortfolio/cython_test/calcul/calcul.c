
int fibo(int n) {
    int x = 0, y = 1, sum, i;
    for (i=0 ; i<n ; i++) {
        sum = x+y;
        x = y;
        y = sum;
    }
    return x;
}

int fibo2(int const n){
    int x = 0, y = 1, sum, i;
    for (i=0 ; i<n ; i++) {
        sum = x+y;
        x = y;
        y = sum;
    }
    return x;

}

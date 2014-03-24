'''
Created on 2014/3/11

@author: chenhh
'''
from hello import say_hello_to, f, f2
import time

def fpy(x):
    return x**2-x


def testf():
    t = time.time()
    for i in xrange(1000000):
        fpy(i)
    print time.time()-t
    
    t= time.time()
    for i in xrange(1000000):
        f(i)
    print time.time()-t
    
    t= time.time()
    for i in xrange(1000000):
        f2(i)
    print time.time()-t

def test2():
    import hello2
    hello2.hello2("chenhh, hello2")

if __name__ == '__main__':
    say_hello_to("Chenhh")
    print f(10)
    testf()
    test2()
    

from cmath import exp
from math import pi
import time
import numpy as np
import random


class FastBigNum0:
    def __init__(self, value):
        self.value = [int(i) for i in reversed(value)]

    def __mul__(self, b):
        size_n = 2*max(len(self.value), len(b.value))
        x_dft = self.dft(self.value, size_n)
        y_dft = self.dft(b.value, size_n)

        z_dft = [x_dft[i]*y_dft[i] for i in range(size_n)]
        z = self.idft(z_dft, size_n)

        ex = 1
        answ = 0
        for i in z:
            answ += i*ex
            ex *= 10

        return FastBigNum0(str(answ))

    def __str__(self):
        return ''.join(str(element) for element in self.value[::-1])

    def omega(self, k, n):
        return exp(-2j*k*pi/n)

    def dft(self, x, n):
        return [sum(x[i]*self.omega(i*k,n) if i<len(x) else 0 for i in range(n)) for k in range(n)]

    def idft(self, x, n):
        return [int(round(sum(x[i]*self.omega(-i*k,n) if i<len(x) else 0 for i in range(n)).real)/n) for k in range(n)]


class FastBigNum1:
    def __init__(self, value):
        self.value = [int(i) for i in reversed(value)]
    
    def __mul__(self, b):
        size_n = 2*max(len(self.value), len(b.value))
        d_len = 1
        while d_len < size_n:
            d_len *= 2
        x = self.value.copy() + [0]*(d_len - len(self.value))
        y = b.value.copy() + [0]*(d_len - len(b.value))
        self.fft(x)
        self.fft(y)

        z = [x[i] * y[i] for i in range(d_len)]
        self.ifft(z)
        z = [int(round(item.real / len(z))) for item in z]
        
        for i in range(1, len(z)):
            z[i] += z[i-1] // 10
            z[i-1] = z[i-1] % 10
        
        for i in range(len(z)-1, 0, -1):
            if z[i] == 0 and z[i-1] != 0:
                z = z[:i]
                break
        
        return FastBigNum1(z[::-1])
    
    def __str__(self):
        return ''.join(str(element) for element in self.value[::-1])

    def omega(self, k, n):
        return exp(-2j*k*pi/n)

    def fft(self, x):
        n = len(x)
        if n <= 1:
            return
        
        even = x[::2]
        odd = x[1::2]
        self.fft(even)
        self.fft(odd)
        for k in range(n//2):
            t = self.omega(k, n) * odd[k]
            x[k] = even[k] + t
            x[n//2 + k] = even[k] - t
            
    def ifft(self, x):
        n = len(x)
        if n <= 1:
            return
        
        even = x[::2]
        odd = x[1::2]
        self.ifft(even)
        self.ifft(odd)
        for k in range(n//2):
            t = self.omega(-k, n) * odd[k]
            x[k] = even[k] + t
            x[n//2 + k] = even[k] - t


class FastBigNum2:
    def __init__(self, value):
        self.value = [int(i) for i in reversed(value)]
    
    def __mul__(self, b):
        size_n = 2*max(len(self.value), len(b.value))
        d_len = 1
        while d_len < size_n:
            d_len *= 2
        x = self.value.copy() + [0]*(d_len - len(self.value))
        y = b.value.copy() + [0]*(d_len - len(b.value))
        x = np.fft.fft(x)
        y = np.fft.fft(y)

        z = np.multiply(x, y)
        z = np.fft.ifft(z)
        z = [int(round(item.real)) for item in z]

        for i in range(1, len(z)):
            z[i] += z[i-1] // 10
            z[i-1] = z[i-1] % 10
        
        for i in range(len(z)-1, 0, -1):
            if z[i] == 0 and z[i-1] != 0:
                z = z[:i]
                break
        
        return FastBigNum1(z[::-1])
    
    def __str__(self):
        return ''.join(str(element) for element in self.value[::-1])


def benchmark():
    for length in [500, 1000, 5000, 10000, 100000, 1000000]:
        a = ''.join([random.choice("0123456789") for i in range(length)])
        b = ''.join([random.choice("0123456789") for i in range(length)])
        
        print("length of sample:", length)

        start = time.process_time()
        int(a)*int(b)
        print('standard mul, test took:', time.process_time() - start)

        X = FastBigNum2(a)
        Y = FastBigNum2(b)
        start = time.process_time()
        Z = X*Y
        print('numpy, test took:', time.process_time() - start)
        assert(str(Z) == str(int(a)*int(b)))
        
        X = FastBigNum1(a)
        Y = FastBigNum1(b)
        start = time.process_time()
        Z = X*Y
        print('FFT, test took:', time.process_time() - start)
        assert(str(Z) == str(int(a)*int(b)))
        
        X = FastBigNum0(a)
        Y = FastBigNum0(b)
        start = time.process_time()
        Z = X*Y
        print('DFT, test took:', time.process_time() - start)
        assert(str(Z) == str(int(a)*int(b)))


if __name__ == "__main__":
    benchmark()
#!/usr/bin/env python3

from math import *

def isPrime(n):
    fsr = int(floor(sqrt(n)))
    if (n % 2) == 0: return False
    for i in range(3, fsr+1, 2):
        if (n % i) == 0: return False
    return True

# Find primes between 3 and 5,000,000
primes = [2]
numPrimes = 1 # Start with 1 for #2
end = 2**23-1

for c in range(3, end+1, 2):
    if isPrime(c):
        print(c)
        #primes.append(c)
        numPrimes = numPrimes + 1

print("Number of primes through", str(end) + ":", numPrimes)
 

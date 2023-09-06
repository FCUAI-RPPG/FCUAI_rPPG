import concurrent.futures
import math


def is_prime(n,m):
    n = n+m
    return n+m
PRIMES = [ i for i in range(1000)]
PRIMES2 = [ j for j in range(100)]
list = []
def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for prime,A in zip(PRIMES, executor.map(is_prime, PRIMES,PRIMES2)):
            list.append(A)
    print(list)
if __name__ == '__main__':
    main()
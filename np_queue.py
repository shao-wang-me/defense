import timeit
from collections import deque

setup = '''
import numpy as np

def push(xs, x):
    xs = np.roll(xs, 1)
    xs[0] = x
    return xs
    
xs = np.random.rand(25)
x = np.random.rand()
'''
stmt = '''
push(xs, x)
'''

print(timeit.timeit(stmt, setup, number=100))
# deque([1], maxlen=100)

setup = '''
import numpy as np
from collections import deque
xs = np.random.rand(25).tolist()
x = np.random.rand()
q = deque(xs, maxlen=25)
'''

stmt = '''
q.append(x)
'''
print(timeit.timeit(stmt, setup, number=100))

setup = '''
import numpy as np

def push(xs, x):
    xs[:-1] = xs[1:]; xs[-1] = x
    return xs

xs = np.random.rand(25).tolist()
x = np.random.rand()
'''

stmt = '''
push(xs, x)
'''
print(timeit.timeit(stmt, setup, number=100))

import numpy as np


def push(xs, x):
    xs[1:] = xs[:-1]
    xs[0] = x


xs = np.arange(10)
print(xs)
x = 10
print(x)
push(xs, x)
print(xs)

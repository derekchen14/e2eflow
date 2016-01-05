from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import pandas

try:
  import keras
except ImportError:
  print "An ImportError with keras was experienced."

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 100

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
# print(f.maker.fgraph.toposort())
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
print "numpy: %s" % numpy.__version__
print "pandas: %s" % pandas.__version__
print "keras: %s" % keras.__version__
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('theano: Used the cpu')
else:
    print('theano: Used the gpu')
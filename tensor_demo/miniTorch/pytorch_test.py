import minitorch
import random

values = [random.randint(1, 10) for _ in range(10)]
t = minitorch.tensor(values)
print(t)
print(t.shape, t._tensor._strides)
tv = t.view(2, 5)
print(tv)
print(tv.shape, tv._tensor._strides)
tvsum = tv.sum(0)
print(tvsum)
print(tvsum.shape, tvsum._tensor._strides)
tvsum = tv.sum()
print(tvsum)
print(tvsum.shape, tvsum._tensor._strides)
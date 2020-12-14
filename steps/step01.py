#step 1
class Variable:
    def __init__(self, data):
        self.data = data
        
        
import numpy as np

# data = np.array(1.0)
# x = Variable(data)
# print(x.data)

# x.data = np.array(2.0)
# print(x.data)


#step 2
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2
        
# x = Variable(np.array(10))
# f = Square()
# y = f(x)
# print(type(f))
# print(y.data)


#step 3
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)


# step4
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# f = Square()
# x = Variable(np.array(2.0))
# dy = numerical_diff(f,x)
# print(dy)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)
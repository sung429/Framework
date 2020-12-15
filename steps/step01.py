import numpy as np

#step 1
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
        
        
#step 2
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        # print(output.creator)
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
        
# x = Variable(np.array(10))
# f = Square()
# y = f(x)
# print(type(f))
# print(y.data)


#step 3, 6
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)

assert y.creator == C
assert y.creator.input == b
assert b.creator == B
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert a.creator == A
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad = B.backward(b.grad)
# x.grad = A.backward(a.grad)
# print(x.grad)

# y.grad = np.array(1.0)
# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)
# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)
# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)
# print(x.grad)

y.grad = np.array(1.0)
y.backward()
print(x.grad)

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
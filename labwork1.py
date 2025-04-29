def f(x):
    return x*x

def f__(x):
    return 2*x

def gradient_descend(L, x):
    for i in range(100):
        x_new = x - L * f__(x)
        print(f'In iteration {i}: x is {x_new} and f(x) is {f(x_new)}')
        x = x_new

L = 0.5
x = 10
gradient_descend(L, x)
import math

file_path = "./loan2.csv"

with open(file_path, "r") as file:
    lines = file.readlines()

data = []
x1_data = []
x2_data = []
y_data = []

for line in lines:
    data.append(line.strip().split(","))

# Remove column name
data.pop(0)

for row in data:
    x1_data.append(float(row[0]))
    x2_data.append(float(row[1]))
    y_data.append(float(row[2]))

# Algorithm implementation
# LOSS FUNCTION
def f(w0, w1, w2, x1, x2, y):
    loss = -y * (w1*x1 + w2*x2 + w0) + math.log(1 + math.exp(w1*x1 + w2*x2 + w0))
    return loss

# SIGMOID FUNCTION
def sigmoid(z):
    return 1/(1 + math.exp(-z))

def f_w0__(w0, w1, w2, x1, x2, y):
    derivative = 1 - y - sigmoid(-(w1*x1 + w2*x2 + w0))
    return derivative

def f_w1__(w0, w1, w2, x1, x2, y):
    derivative = -y*x1 + x1*(1 - sigmoid(-(w1*x1 + w2*x2 + w0)))
    return derivative

def f_w2__(w0, w1, w2, x1, x2, y):
    derivative = -y*x2 + x2*(1 - sigmoid(-(w1*x1 + w2*x2 + w0)))
    return derivative

def gradient_descend(r, w0, w1, w2, x1_data, x2_data, y_data, threshold):
    n = len(data)
    i = 0
    while True:
        derivative_w0 = sum(f_w0__(w0, w1, w2, x1_data[j], x2_data[j], y_data[j]) for j in range(n)) / n
        derivative_w1 = sum(f_w1__(w0, w1, w2, x1_data[j], x2_data[j], y_data[j]) for j in range(n)) / n
        derivative_w2 = sum(f_w2__(w0, w1, w2, x1_data[j], x2_data[j], y_data[j]) for j in range(n)) / n
        new_w0 = w0 - r * derivative_w0
        new_w1 = w1 - r * derivative_w1
        new_w2 = w2 - r * derivative_w2
        new_f = sum(f(new_w0, new_w1, new_w2, x1_data[j], x2_data[j], y_data[j]) for j in range(n)) / n
        print(f'In iteration {i}: w0 is {w0}, w1 is {w1}, w2 is {w2}, and f(w0, w1, w2) is {new_f}')
        if (new_f < threshold):
            break
        else:
            i = i + 1
            w0 = new_w0
            w1 = new_w1
            w2 = new_w2

if __name__ == "__main__":
    gradient_descend(r=0.01, w0=0, w1=1, w2=2, x1_data=x1_data, x2_data=x2_data, y_data=y_data, threshold=0.1)
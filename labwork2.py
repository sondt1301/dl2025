file_path = "./lr.csv"

with open(file_path, "r") as file:
    lines = file.readlines()

data = []
x_data = []
y_data = []

for line in lines:
    data.append(line.strip().split(","))

for row in data:
    x_data.append(int(row[0]))
    y_data.append(int(row[1]))

# Algorithm implementation
def f(w0, w1, x, y):
    loss = 1/2 * (w1 * x + w0 -y)**2
    return loss

def f_w0__(w0, w1, x, y):
    derivative = w1 * x + w0 - y
    return derivative

def f_w1__(w0, w1, x, y):
    derivative = x * (w1 * x + w0 - y)
    return derivative

def gradient_descend(r, w0, w1, x_data, y_data, threshold):
    n = len(data)
    i = 0
    while True:
        derivative_w0 = sum(f_w0__(w0, w1, x_data[j], y_data[j]) for j in range(n)) / n
        derivative_w1 = sum(f_w1__(w0, w1, x_data[j], y_data[j]) for j in range(n)) / n
        new_w0 = w0 - r * derivative_w0
        new_w1 = w1 - r * derivative_w1
        new_f = sum(f(new_w0, new_w1, x_data[j], y_data[j]) for j in range(n)) / n
        print(f'In iteration {i}: w0 is {w0}, w1 is {w1}, and f(w0, w1) is {new_f}')
        if (new_f < threshold):
            break
        else:
            i = i + 1
            w0 = new_w0
            w1 = new_w1

if __name__ == "__main__":
    gradient_descend(r=0.0001, w0=0, w1=1, x_data=x_data, y_data=y_data, threshold=0.01)
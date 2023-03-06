import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
print(data)

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        # total error formular
        total_error += (y - (m * x + b)) ** 2

    return total_error / float(len(points))


def gradient_descent(m_now, b_now, points, l):
    m_gradient = 0
    b_gradient = 0

    n = len(points)
    loss = loss_function(m_now, b_now, points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * l
    b = b_now - b_gradient * l

    return m, b, loss


m = 0
b = 0
l = 0.0001
epochs = 300

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epochs:{i}")
    m, b, loss = gradient_descent(m, b, data, l)
    print(f"Loss: {loss}")

# calculate R-squared value
y_mean = data.score.mean()
data['y_pred'] = m * data.studytime + b
tss = ((data.score - y_mean) ** 2).sum()
rss = ((data.score - data.y_pred) ** 2).sum()
r_squared = 1 - (rss / tss)
print(f"R-squared: {r_squared}")

plt.scatter(data.studytime, data.score, color="black")
plt.plot(list(range(20, 80)), [m * x + b for x in range(20, 80)], color="red")
plt.show()

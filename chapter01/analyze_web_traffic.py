# -*- coding: utf-8 -*-

import scipy as sp
import matplotlib.pyplot as plt


def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)


data = sp.genfromtxt('chapter01/data/web_traffic.tsv', delimiter='\t')

x = data[:, 0]
y = data[:, 1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

plt.scatter(x, y)
plt.title('Web traffic over the last month')
plt.xlabel('Time')
plt.ylabel('Hits/hour')
plt.xticks([w * 7 * 24 for w in range(10)], ['week {}'.format(w) for w in range(10)])
plt.autoscale(tight=True)
plt.ylim(ymin=0)
plt.grid()

fx = sp.linspace(0, x[-1], 1000)
legends = []

# fit model
print("=== fit model ===")
for i in [1, 2, 3, 10, 50]:
    f = sp.poly1d(sp.polyfit(x, y, i))
    print('Error d={}: {}'.format(i, error(f, x, y)))

    plt.plot(fx, f(fx), linewidth=4)
    legends.append("d={}".format(f.order))

# inflection point
inflection = 3.5 * 7 * 24
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

print("=== inflection point ===")
print('Error inflection: {}'.format(error(fa, xa, ya) + error(fb, xb, yb)))
plt.plot(fx, fa(fx), linewidth=4)
plt.plot(fx, fb(fx), linewidth=4)

# separate train data from test data
frac = 0.3
split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])

print("=== separate train data from test data ===")
for i in [1, 2, 3, 10, 50]:
    f = sp.poly1d(sp.polyfit(xb[train], yb[train], i))
    print('Error d={}: {}'.format(i, error(f, xb[test], yb[test])))

# predict
print("=== predict ===")
fb2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
print(fb2)
print(fb2 - 100000)
from scipy.optimize import fsolve
reached_max = fsolve(fb2 - 100000, 800) / (7 * 24)
print('100,000 hits/hour expected at week {}'.format(reached_max[0]))

plt.legend(legends, loc="upper left")
plt.show()


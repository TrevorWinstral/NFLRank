import matplotlib.pyplot as plt
import numpy as np

def logistic10(x, beta):
    return 1/(1+10**(x*beta))

xlim = 28
beta = -0.1
t = np.linspace(-xlim, xlim, 4000)
plt.plot(t, logistic10(t, beta), c='black')
xtick_locs = [i for i in range(-xlim, xlim+1, 7)]
xtick_labels = [str(i) for i in xtick_locs]
plt.xticks(xtick_locs, xtick_labels)
plt.xlabel('Point Differential')
plt.ylabel('Realized Score')

#dashed lines for
x=0
xx = [-xlim,x]
plt.plot(xx, [logistic10(x, beta)]*2, c='black', linestyle='--')
plt.plot([x ,x], [0, logistic10(x, beta)], c='black', linestyle='--')

x=7
xx = [-xlim,x]
plt.plot(xx, [logistic10(x, beta)]*2, c='black', linestyle='--')
plt.plot([x ,x], [0, logistic10(x, beta)], c='black', linestyle='--')

plt.savefig('score_curve.svg')
plt.show()
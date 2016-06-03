import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt


epochs = [ii for ii in range(1, 51, 1)]
LL = 7  # Valid LL: 7, EMF: 9, MSE: 11

# TODO - change it in the morning!
def plot(filename):
    temp = np.zeros((len(epochs), 4))
    for ii in range(1, 5, 1):
        file = filename % ii
        data = genfromtxt(file, skip_header=True, delimiter=',')
        temp[:, ii-1] = data[:, LL]

    sd = np.std(temp, axis=1)
    mean = np.mean(temp, axis=1)
    return mean, sd


# Proxy LL on validation set
LL = 7
fig, ax = plt.subplots(1)
ax.axis([0, 50, -.13, 0])
file = '../scripts/CD-1_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'CD')
file = '../scripts/MF-3_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'MF-3')
file = '../scripts/TAP2-3_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'TAP2-3')
file = '../scripts/CD-10_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'CD-10')
file = '../scripts/MF-10_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'MF-10')
file = '../scripts/TAP2-10_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'TAP2-10')
file = '../scripts/TAP3-3_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'TAP3-3')
ax.legend(loc=4)
plt.savefig('../scripts/validLL.pdf')

# EMF on validation set
LL = 9
fig, ax = plt.subplots(1)
ax.axis([0, 50, -.13, 0])
file = '../scripts/CD-1_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'CD')
file = '../scripts/MF-3_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'MF-3')
file = '../scripts/TAP2-3_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'TAP2-3')
file = '../scripts/CD-10_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'CD-10')
file = '../scripts/MF-10_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'MF-10')
file = '../scripts/TAP2-10_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'TAP2-10')
file = '../scripts/TAP3-3_%s.csv'
mean, sd = plot(file)
ax.errorbar(epochs[5:], mean[5:], yerr=sd[5:], label=r'TAP3-3')
ax.legend(loc=4)
plt.savefig('../scripts/validEMF.pdf')




import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoLocator

data = [-0.95, -0.95, -0.9, -0.85, -0.7, -0.6, -0.55, -0.45, -0.35, -0.25, -0.1, 0.0,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.85, 0.95]
data2 = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

bins = np.arange(-1.0, 1.1, 0.10)
hist, _ = np.histogram(data, bins=bins)

plt.bar(bins[:-1], hist, width=0.05, align='edge')
plt.xlabel('Value Range')
plt.ylabel('Data Count')
plt.title('Data Distribution')
plt.xticks(bins + 0.05, rotation=45, ha='right')
plt.gca().yaxis.set_major_locator(AutoLocator())
total = sum(hist)
for i, v in enumerate(hist):
    plt.text(bins[i] + 0.04, v, str(v) + ' ({:.1f}%)'.format(v/total*100), fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))
plt.savefig('data_distribution.png')

hist2, _ = np.histogram(data2, bins=bins)
plt.bar(bins[:-1], hist2, width=0.05, align='edge')
plt.xlabel('Value Range')
plt.ylabel('Data Count')
plt.title('Data Distribution')
plt.xticks(bins + 0.05, rotation=45, ha='right')
plt.gca().yaxis.set_major_locator(AutoLocator())
total = sum(hist2)
for i, v in enumerate(hist2):
    plt.text(bins[i] + 0.04, 1.05 * v, str(v) + ' ({:.1f}%)'.format(v/total*100), fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))
plt.savefig('data_distribution2.png')
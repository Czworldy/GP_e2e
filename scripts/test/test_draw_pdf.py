
import seaborn as sns
import matplotlib.pyplot as plt


x = [1,2,3]
y = [-1, 0, 1]

sns.kdeplot(x, y, shade=True, bw="silverman", gridsize=50, clip=(-11, 11),  cmap="Purples")

with sns.axes_style('white'):
    # sns.jointplot('X', 'Y', data, kind='kde')
    sns.jointplot(x, y, kind='kde')
plt.show()
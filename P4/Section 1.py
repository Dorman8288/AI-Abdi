import numpy as np
import numpy.random as rand
from sklearn import svm
import matplotlib.pylab as plt
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.widgets import Slider

def exp(x, y):
    return np.exp(x) < y

def sin(x, y):
    return np.sin(x) < y

def sinc(x, y):
    return np.sinc(x) < y

def ran(x, y):
    return x ** 2 + y ** 5 < x ** 2 - np.cos(y)

def Circle(x, y):
    return x ** 2 + y ** 2 <= 1

def Linear(x, y):
    return x + y + 2 <= 1

def GenerateRandomPoints(xrange, yrange, count):
    x = rand.uniform(xrange[0], xrange[1], count)
    y = rand.uniform(yrange[0], yrange[1], count)
    return x, y


function = Circle
count = 500
xrange = (-2, 2)
yrange = (-2, 2)
kernel = "rbf"
cache_Size = 700

gamma = 0.5
C = 0.5
degree = 1
coef = 10

points_x, points_y = GenerateRandomPoints(xrange, yrange, count)
colors = ["blue" if function(points_x[i], points_y[i]) else "red" for i in range(count)] 


X = np.array([[points_x[i], points_y[i]] for i in range(count)])
y = np.array([function(X[i][0], X[i][1]) for i in range(count)])

fig, ax = plt.subplots()
ax.set_xlim([xrange[0], xrange[1]])
ax.set_ylim([yrange[0], yrange[1]])
ax.scatter(points_x, points_y, c=colors)

fig.subplots_adjust(left=0.25, bottom=0.25)

axC = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
C_slider = Slider(
    ax=axC,
    label="C",
    valmin=0,
    valmax=1,
    valinit=C,
    orientation="vertical"
)

if kernel == "rbf":
    gamma = 0.5
    axgamma = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    gamma_slider = Slider(
        ax=axgamma,
        label='Gamma',
        valmin=0,
        valmax=2,
        valinit=gamma,
    )
elif kernel == "linear":
    pass
elif kernel == "poly":
    degree = 1
    axdegree = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    degree_slider = Slider(
        ax=axdegree,
        label='degree',
        valmin=0,
        valmax=20,
        valinit=degree,
        valstep=1
    )
    coef = 1
    axcoef = fig.add_axes([0.248, 0.15, 0.65, 0.03])
    coef_slider = Slider(
        ax=axcoef,
        label='coef',
        valmin=0,
        valmax=20,
        valinit=coef,
    )

def change_degree(val):
    ax.clear()
    degree = val
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef, cache_size=cache_Size)
    model.fit(X, y)
    
    ax.scatter(points_x, points_y, c=colors)
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=ax,
    )
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    ax.set_xlim([xrange[0], xrange[1]])
    ax.set_ylim([yrange[0], yrange[1]])


def change_coef(val):
    ax.clear()
    coef = val
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef, cache_size=cache_Size)
    model.fit(X, y)
    
    ax.scatter(points_x, points_y, c=colors)
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=ax,
    )
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    ax.set_xlim([xrange[0], xrange[1]])
    ax.set_ylim([yrange[0], yrange[1]])



def change_gamma(val):
    ax.clear()
    gamma = val
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef, cache_size=cache_Size)
    model.fit(X, y)
    
    ax.scatter(points_x, points_y, c=colors)
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=ax,
    )
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    ax.set_xlim([xrange[0], xrange[1]])
    ax.set_ylim([yrange[0], yrange[1]])

def change_C(val):
    ax.clear()
    C = val
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef, cache_size=cache_Size)
    model.fit(X, y)
    
    ax.scatter(points_x, points_y, c=colors)
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=ax,
    )
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    ax.set_xlim([xrange[0], xrange[1]])
    ax.set_ylim([yrange[0], yrange[1]])

if kernel == "rbf":
    gamma_slider.on_changed(change_gamma)
if kernel == "poly":
    degree_slider.on_changed(change_degree)
    coef_slider.on_changed(change_coef)
C_slider.on_changed(change_C)
plt.show()
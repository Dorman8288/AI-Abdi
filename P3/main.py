from ExpressionTree import *
import math
import matplotlib.pyplot as plt

tree = ExpressionTree(
    ["x"],
    [
        ("+", lambda x, y: x + y),
        ("-", lambda x, y: x - y),
        ("/", lambda x, y: x / y),
        ("*", lambda x, y: x * y),
        ("^", lambda x, y:  math.pow(x, y))

    ],
    [
        ("sin", math.sin),
        ("cos", math.cos)
    ],
    (-3, 3),
    False,
    0.1
)
tree.MakeRandom(0.999, 0.8)


def GenerateRandomPopulation(configs, treeSetup):
    population = []
    for degeradationRate, count in configs:
        for _ in range(count):
            tree = copy.deepcopy(treeSetup)
            tree.MakeRandom(0.999, degeradationRate)
            population.append(tree)
    return population

xrange = (-20, 20)
yrange = (-10, 10)
function = lambda x: x ** 2 + 3*x + 10

population = GenerateRandomPopulation([(0.8, 10)], tree)
for tree in population:
    print(tree.display())
# DataPoints = np.linspace(xrange[0], xrange[1], (xrange[1] - xrange[0]) * 10)
# correct = [function(x) for x in DataPoints]
# values = [tree.evalueate({"x": x}) for x in DataPoints]
# plt.ylim(yrange[0], yrange[1])
# plt.xlim(xrange[0], xrange[1])
# plt.plot(DataPoints, values, color="red")
# plt.plot(DataPoints, correct, color="blue")
# plt.show()
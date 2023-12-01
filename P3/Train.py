from ExpressionTree import *
import math
import matplotlib.pyplot as plt

tree = ExpressionTree(
    ["x"],
    [
        ("+", lambda x, y: x + y),
        ("-", lambda x, y: x - y),
        ("/", lambda x, y: x / y),
        ("*", lambda x, y: x * y)
    ],
    [
        ("sin", math.sin),
        ("cos", math.cos)
    ],
    (-3, 3),
    True,
    0.1,
    1,
    0.8,
    300
)
tree.MakeRandom(0.999, 0.95)
print(tree.display())

def GenerateRandomPopulation(configs, treeSetup):
    population = []
    for degeradationRate, count in configs:
        for _ in range(count):
            while True:
                tree = copy.deepcopy(treeSetup)
                tree.MakeRandom(0.999, degeradationRate)
                if type(tree.root) != StaticNode:
                    break
            population.append(tree)
    return population

def Train(initialPopulation, target):
    i = 0
    minLoss = math.inf
    bestTree = None
    for tree in initialPopulation:
        print(i)
        i += 1
        try:
            tree.optimizeStatic(target)
            loss = tree.SqueredLoss(target)
            if minLoss > loss:
                print("yes")
                minLoss = loss
                bestTree = tree
        except:
            initialPopulation.remove(tree)

    
        
xrange = (-10, 10)
yrange = (-500000, 500000)
function = lambda x: (26 + x) * x * -1000
datapoints = np.linspace(xrange[0], xrange[1], 10)
target = [({"x": x}, function(x)) for x in datapoints]

initialPopulation = GenerateRandomPopulation([(0.6, 200), (0.9, 100), (0.95, 50)], tree)
#print([tree.display() for tree in initialPopulation])
Train(initialPopulation, target)



# plt.ylim(yrange[0], yrange[1])
# plt.xlim(xrange[0], xrange[1])
# plt.plot(datapoints, valuesBeforeOptim, color="red")
# plt.plot(datapoints, valuesAfterOptim, color="green")
# plt.plot(datapoints, correct, color="blue")
# plt.show()
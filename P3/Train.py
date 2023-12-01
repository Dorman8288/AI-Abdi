from ExpressionTree import *
import math
import matplotlib.pyplot as plt
from queue import PriorityQueue
import copy

tree1 = ExpressionTree(
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
tree2 = ExpressionTree(
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


def reproduce(population, mutationChance):
    children = []
    for parent1 in population:
        for parent2 in population:
            if parent1 != parent2:
                a, b = ExpressionTree.crossover(parent1, parent2)
                chance = np.random.uniform(0, 1)
                if chance < mutationChance:
                    a.Evolve()
                children.append(a)
                if chance < mutationChance:
                    b.Evolve()
                children.append(b)
    return population

def K_Best(candidates, k):
    queue = PriorityQueue(k)
    for item in candidates:
        try:
            #print(item)
            if queue.qsize() < k:
                queue.put(item)
                continue
            #print(min)
            min = queue._get()[0]
            best = item if item[0] > min else min
            queue.put(best)
        except:
            continue
    return queue.queue

def Train(initialPopulation, target, PickingStrategy, k, mutationChance):
    population = initialPopulation
    while True:
        i = 0
        candidates = []
        print("hello")
        for tree in population:
            i += 1
            print(i)
            try:
                tree.optimizeStatic(target)
                loss = tree.SqueredLoss(target)
                #print(loss, tree.display())
                candidates.append((loss, tree))
            except:
                initialPopulation.remove(tree)
        winners = PickingStrategy(candidates, k)
        bestloss = winners[0][0]
        print("Loss:", bestloss)
        if bestloss < 0.01:
            return winners[0][1]
        #print(winners)
        temp = []
        for item in winners:
            if type(item) != float:
                temp.append(tree)
        children = reproduce(temp, mutationChance)
        print(children)
        print(len(children))
        j = 0
        for i in range(len(children)):
            temp.append(children[i])
            print(children[i].display())
            j += 1
        population = temp

    
        
xrange = (-3, 3)
yrange = (-500000, 500000)
function = lambda x: x * x
datapoints = np.linspace(xrange[0], xrange[1], 10)
target = [({"x": x}, function(x)) for x in datapoints]

tree1.MakeRandom(0.999, 0.8)
tree2.MakeRandom(0.999, 0.8)
print("A: ", tree1.display())
print()
print("B: ", tree2.display())
print()
a, b = ExpressionTree.crossover(tree1, tree2)
print("A: ", tree1.display())
print()
print("B: ", tree2.display())
print()
print("C: ", a.display())
print()
print("D: ", b.display())

#initialPopulation = GenerateRandomPopulation([(0.6, 20), (0.9, 10), (0.95, 5)], tree)
#print([tree.display() for tree in initialPopulation])
#Train(initialPopulation, target, K_Best, 20, 0.2)




# plt.ylim(yrange[0], yrange[1])
# plt.xlim(xrange[0], xrange[1])
# plt.plot(datapoints, valuesBeforeOptim, color="red")
# plt.plot(datapoints, valuesAfterOptim, color="green")
# plt.plot(datapoints, correct, color="blue")
# plt.show()
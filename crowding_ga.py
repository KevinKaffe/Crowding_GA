from LinReg import LinReg
from sklearn import datasets
import numpy as np
import random
import math
import matplotlib.pyplot as plt
linear_regression = LinReg()

# Parameters
# MC = Mutation chance, PHI= Generalized crowding parameter
MC = 0.7
PHI = 0.3

def read_data():
    rows = []
    ys = []
    f = open('datad.txt')
    for line in f:
        tokens = line.split()
        tokens = [float(token.replace(' ', '').replace('\n', '')) for token in tokens]
        rows.append(tokens[:-1])
        ys.append(tokens[-1])
    return np.asarray(rows), ys

x,y = read_data()


class Chromosome():
    def __init__(self, bitstring):
        self.bitstring = bitstring
        self.fitness = 0

def get_fitness(individual):
    global x
    if individual.fitness!=0:
        return individual.fitness
    selected_features = linear_regression.get_columns(x, individual.bitstring)
    individual.fitness = 1/linear_regression.get_fitness(selected_features,y)
    return individual.fitness
    
def generate_population(pop_size, i_size):
    # Generates a population of pop_size chromosomes containing bitstrings of size i_size
    pop = []
    for i in range(pop_size):
        c = Chromosome("{0:b}".format(random.getrandbits(i_size)).zfill(i_size))
        pop.append(c)
    return pop

def selection(pop, parent_amount):
    fitness_scores = []
    for i in pop:
        fitness_scores.append(get_fitness(i))
    # Calculate scaled scores
    scores = np.asarray(fitness_scores) - min(fitness_scores)*0.99
    scores/=(np.sum(scores)+1e-20)
    parent_list = []
    parent_score = list(zip(pop, scores))
    # Select parents with probability of their score
    for i in range(parent_amount):
        random_variable = random.random()
        for parent, score in parent_score:
            random_variable-=score
            if random_variable<0:
                parent_list.append(parent)
                break
    return parent_list

def find_best(c1,c2):
    # Generalized crowding tournament
    f1 = get_fitness(c1)
    f2 = get_fitness(c2)
    r = random.random()
    if f1>f2:
        if f1/(f1+PHI*f2) >= r:
            return c1
        else:
            return c2
    else:
        if PHI*f1/(PHI*f1+f2) >= r:
            return c1
        else:
            return c2
    
def d(b1,b2):
    # Simple bitstring distance
    counter = 0
    for i in range(len(b1)):
        if b1[i] == b2[i]:
            counter +=1
    return counter

def local_tourn(c1,c2,p1,p2):
    d11 = d(c1.bitstring, p1.bitstring)
    d22 = d(c2.bitstring, p2.bitstring)

    d21 = d(c2.bitstring, p1.bitstring)
    d12 = d(c1.bitstring, p2.bitstring)

    if d11+d22 < d21+d12:
        r1 = find_best(c1,p1)
        r2 = find_best(c2,p2)
    else:
        r1 = find_best(c1,p2)
        r2 = find_best(c2,p1)
    return r1, r2
    
def crossover(p1,p2, crowding=False):
    crossover_point = random.randint(0, len(p1.bitstring))
    c1_value = p1.bitstring[:crossover_point] + p2.bitstring[crossover_point:]
    c2_value = p2.bitstring[:crossover_point] + p1.bitstring[crossover_point:]
    c1_value = mutation(c1_value)
    c2_value = mutation(c2_value)
    c1 = Chromosome(c1_value)
    c2 = Chromosome(c2_value)
    if crowding:
        c1,c2 = local_tourn(c1,c2,p1,p2)
    return c1,c2

def mutation(c):
    r = random.random()
    if r<MC:
        m_index = random.randint(0, len(c)-1)
        c = list(c)
        if c[m_index]=='1':
            c[m_index] = '0'
        else:
            c[m_index] = '1'
        r = random.random()
    return "".join(c)

def mating(parents, crowding=False):
    children = []
    random.shuffle(parents)
    for i in range(int(len(parents)/2)):
        p1 = (parents[2*i])
        p2 = (parents[2*i+1])
        c1, c2=crossover(p1,p2, crowding=crowding)
        children.extend((c1,c2))
    return children

def survival(pop, pop_size):
    fitness_scores = []
    for i in pop:
        fitness_scores.append(get_fitness(i))
    scores = np.asarray(fitness_scores)
    parents = list(zip(pop, scores))
    parents.sort(key=lambda x: x[1])
    parents.reverse()
    parents = parents[:pop_size]
    return [parent[0] for parent in parents]

def get_entropy(bitstrings):
    probabilities = []
    for i in range(len(bitstrings[0])):
        probabilities.append(0)
        for bitstring in bitstrings:
           probabilities[i]+=float(int(bitstring[i]))/len(bitstrings)
    entropy=0
    for pi in probabilities:
        if pi!=0:
            entropy-=pi*math.log(pi,2)
    return entropy
        
def pop_entropy(population):
    bitstrings = [ind.bitstring for ind in population]
    return get_entropy(bitstrings)
            
def run_genetic(population, generations, crowding=False):
    pop = generate_population(population, x.shape[1])
    fitness_progression= []
    entropies = []
    for i in range(generations):
        for ind in pop:
            ind.fitness=0
        print('Generation: ', i)
        if crowding:
            parents = selection(pop, parent_amount=population)
            children = mating(parents, crowding=True)
            pop =survival(children, population)
        else:
            parents = selection(pop, parent_amount=int(population/2))
            children = mating(parents)
            pop =survival(pop+children, population)
            
        entropy = pop_entropy(pop)
        entropies.append(entropy)
        fitness_progression.append(pop[0].fitness)
    return fitness_progression, entropies

print('Running crowding')
fitness_crowding, entropy_crowding = run_genetic(100,200,crowding=True)
print('Running simple genetic algorithm')
fitness_simple, entropy_simple = run_genetic(100,200,crowding=False)

averaged_fitness_crowding = []
fitness_crowding = [fitness_crowding[0],fitness_crowding[0]] + fitness_crowding + [fitness_crowding[-1],fitness_crowding[-1]]
for i in range(2, len(fitness_crowding)-2):
    averaged_fitness_crowding.append(np.sum(fitness_crowding[i-2:i+3])/5)

averaged_fitness_simple = []
fitness_simple = [fitness_simple[0],fitness_simple[0]] + fitness_simple + [fitness_simple[-1],fitness_simple[-1]]
for i in range(2, len(fitness_simple)-2):
    averaged_fitness_simple.append(np.sum(fitness_simple[i-2:i+3])/5)
plt.plot(averaged_fitness_crowding, label='Crowding')
plt.plot(averaged_fitness_simple, label='Simple Genetic Algorithm')
plt.legend(loc='best')
plt.show()

plt.plot(entropy_crowding, label='Crowding')
plt.plot(entropy_simple, label='Simple Genetic Algorithm')
plt.legend(loc='best')
plt.show()

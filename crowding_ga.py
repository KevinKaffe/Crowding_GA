from LinReg import LinReg
from sklearn import datasets
import numpy as np
import random
import math
import matplotlib.pyplot as plt
linear_regression = LinReg()

# Parameters
# MC = Mutation chance, PHI= Generalized crowding parameter
MC = 0.0
MR = 0.3
PHI = 0.0
CP = 0.7
FITNESS = "SINE"
BITSTRING_SIZE=20

def read_data():
    rows = []
    ys = []
    f = open('new.data')
    for line in f:
        tokens = line.split(',')
        tokens = [token.replace(' ', '').replace('\n', '') for token in tokens]
        rows.append(tokens[:-1])
        ys.append(float(tokens[-1]))
    rows = np.asarray(rows)
    mask = (rows=='?')
    idx = mask.any(axis=0)
    rows = rows[:,~idx]
    rows = rows.astype('float')
    for i in range(len(idx)):
        if idx[i]:
            if i>3:
                print(str(i+1) + ',', end='')
            else:
                print(str(i) + ',', end='')
    f.close()
    return rows, ys

def save_data(x,y):
    f = open('new.data', 'w')
    for i in range(len(x)):
        line = ''
        for j in range(len(x[i])):
            line+=str(x[i][j])+','
        line+=str(y[i])
        f.write(line + '\n')
    f.close()

x,y = read_data()
print(1/linear_regression.get_fitness(x,y, random_state=42))
#save_data(x,y)
print(x.shape)
class Chromosome():
    def __init__(self, bitstring):
        self.bitstring = bitstring
        self.fitness = 0

def f(x):
    x_t = x/(10*(2**10))
    return math.sin(x_t)

def get_fitness(individual):
    global x
    if individual.fitness!=0:
        return individual.fitness
    if FITNESS=="DATASET":
        selected_features = linear_regression.get_columns(x, individual.bitstring)
        individual.fitness = 1/linear_regression.get_fitness(selected_features,y, random_state=42)
    elif FITNESS=="SINE":
        individual.fitness = f(int(individual.bitstring,2))
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
    scores = np.asarray(fitness_scores)
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
        counter+=(int(b1[i])-int(b2[i]))**2
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
    c1_value = ''
    for i in range(len(p1.bitstring)):
        r = random.random()
        if r>CP:
            c1_value+=p1.bitstring[i]
        else:
            c1_value+=p2.bitstring[i]
    c2_value = ''
    for i in range(len(p2.bitstring)):
        r = random.random()
        if r<CP:
            c2_value+=p1.bitstring[i]
        else:
            c2_value+=p2.bitstring[i]
    
    c1_value = mutation(c1_value)
    c2_value = mutation(c2_value)
    c1 = Chromosome(c1_value)
    c2 = Chromosome(c2_value)
    if crowding:
        c1,c2 = local_tourn(c1,c2,p1,p2)
    return c1,c2

def old_mutation(c):
    c = list(c)
    r = random.random()
    if r>MC:
        return "".join(c)
    for i in range(0,len(c)):
        m_index = i
        r = random.random()
        if r<MR:

            if c[m_index]=='1':
                c[m_index] = '0'
            else:
                c[m_index] = '1'
            r = random.random()
    return "".join(c)

def mutation(c):
    m_index = random.randint(0, len(c)-1)
    c = list(c)
    r = random.random()
    if r>MC:
        return "".join(c)
        if c[m_index]=='1':
            c[m_index] = '0'
        else:
            c[m_index] = '1'
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

def crowding_survival(children, parents):
    new_pop = []
    random.shuffle(children)
    random.shuffle(parents)
    for i in range(int(len(parents)/2)):
        p1,p2 = local_tourn(children[2*i], children[2*i+1], parents[2*i], parents[2*i+1])
        new_pop.append(p1)
        new_pop.append(p2)
    return new_pop

def genetic_insert(population, insertions):
    for i in insertions:
        scores = []
        for p in population:
            scores.append(d(i.bitstring,p.bitstring))
        closest = np.argmin(scores)
        if population[closest].fitness < i.fitness:
            population[closest] = i
    return population

def plot_sin(pop):
    time = np.arange(0,128,0.1)
    amp = np.sin(time)
    x=[]
    y=[]
    for p in pop:
        x.append(int(p.bitstring,2)/(10*(2**10)))
        y.append(f(int(p.bitstring,2)))
    plt.scatter(x,y, c='orange')
    plt.plot(time, amp)
    plt.title("Population plot")
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.show()
    
def run_genetic(population, generations, crowding=False):
    genotype_size=0
    if FITNESS=="SINE":
        genotype_size = BITSTRING_SIZE
    elif FITNESS=="DATASET":
        genotype_size= x.shape[1]
    pop = generate_population(population, genotype_size)
    best = []
    averages = []
    entropies = []
    for i in range(generations):
        if i%10 == 0:
            print('Generation %d/%d' % (i, generations))
        if crowding:
            parents = pop
            children = mating(parents, crowding=True)
            elitism = survival(pop, 2)
            pop = genetic_insert(children, elitism)
            random.shuffle(pop)
        else:
            parents = selection(pop, parent_amount=population)
            children = mating(parents)
            pop =survival(pop+children, population)
        if FITNESS=="SINE" and i%10==0:
            plot_sin(pop)
        entropy = pop_entropy(pop)
        entropies.append(entropy)
        fitnesses = [ind.fitness for ind in pop]
        best.append(np.max(fitnesses))
        averages.append(np.average(fitnesses))
    return best, averages, entropies


print('Running crowding')
crowding_best, crowding_avg, entropy_crowding = run_genetic(70,85,crowding=True)
print('Running simple genetic algorithm')
simple_best, simple_avg, entropy_simple = run_genetic(70,85,crowding=False)

plt.plot(crowding_best, label="Deterministic Crowding")
plt.plot(simple_best, label="Simple genetic algorithm")
plt.legend(loc="best")
plt.title("Best fitness")
plt.show()

plt.plot(crowding_avg, label="Deterministic Crowding")
plt.plot(simple_avg, label="Simple genetic algorithm")
plt.legend(loc='best')
plt.title("Average fitness")
plt.show()

plt.plot(entropy_crowding, label='Deterministic Crowding')
plt.plot(entropy_simple, label='Simple Genetic Algorithm')
plt.legend(loc='best')
plt.title("Entropy measure")
plt.show()

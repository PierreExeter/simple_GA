'''
Implementation of a single objective genetic algorithm with 2 design paramaters (x and y)
Minimise the function Z(x, y) = x*sin(x)*y*cos(y)
def [A, Ao, best, best_o] = GA(nb_params, pop_size, nb_gen, p_mut) 
'''

import numpy as np
import matplotlib.pyplot as plt

# optimisation parameters
nb_params = 2                           # nb of design parameters
pop_size = 50                           # population size. It should be an even number, otherwise the crossover might not work
nb_gen = 50                             # nb of generations
p_mut = 0.2                             # probability of mutations
nb_mutant = int(round(pop_size*p_mut))  # nb of mutants in the current child population

# min and max value of the design parameters
min1 = 0
max1 = 20
min2 = 0
max2 = 20 

# cost function
x = np.linspace(min1, max1, 100)
X, Y = np.meshgrid(x, x)
Z = X*np.sin(X)*Y*np.cos(Y)

# random initial population with design parameters betweeen 0 and 1
A = np.random.rand(pop_size, nb_params)

# scale the initial population's design parameters between a max and min value
A[:, 0] = min1+max1*A[:, 0]
A[:, 1] = min2+max2*A[:, 1]

# evaluate the objective
Ao = np.zeros(pop_size)
for i in range(pop_size):
    Ao[i] = A[i, 0]*np.sin(A[i, 0])*A[i, 1]*np.cos(A[i, 1])

# sort A and Ao by ascending number of objective fitness
A_ind = np.argsort(Ao)
Ao = Ao[A_ind]
A = A[A_ind, :]

# create best list
best_list = []
gen_list = []
best_list.append(Ao[0])
gen_list.append(0)

# plot the search space + initial population
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
levels = np.arange(-300, 300, 30)
ax1.plot(A[:, 0], A[:, 1], 'ro')
ax1.plot(A[0, 0], A[0, 1], 'ko', markersize=10)  # the best individual is the first of the list
ax1.contour(X, Y, Z, levels=levels)
ax1.set_xlabel("design parameter x")
ax1.set_ylabel("design parameter y")
ax1.set_title("design space")

# plot objective spance
ax2.plot(gen_list, best_list)
ax2.set_xlabel("Number of generations")
ax2.set_ylabel("Best objective value")
ax2.set_title("objective space")

plt.pause(0.1)
plt.ion()


for q in range(nb_gen):
    
    # selection: only keep the best half of the population (i.e. the breeding parent population A_p)
    A_p = A[:pop_size//2, :]
    Ao_p = Ao[:pop_size//2]

    # crossover: create a new child from 2 parents 
    # The child's parameters are located randomly in a rectangle whose opposite corners are the 2 parents
    A_c = np.zeros((pop_size//2, nb_params))  # initialise
    
    for i in range(pop_size//2):
        # pick 2 random individuals in the parent population 
        parent_pool = np.random.permutation(A_p)
        p1 = parent_pool[0, :]
        p2 = parent_pool[1, :]
        # child created from the 2 parents
        A_c[i, 0] = min(p1[0],p2[0])+np.random.rand(1)*abs(p1[0]-p2[0]) # param 1
        A_c[i, 1] = min(p1[1],p2[1])+np.random.rand(1)*abs(p1[1]-p2[1]) # param 2
    
    # mutation: modify the children by changing randomly the objectives of p_mut percent of the parent population
    # pick nb_mutant random children 
    ind = np.random.permutation(pop_size//2)  # index list of the children to be mutated (the first nb_mutant individuals will be affected)
    for i in range(nb_mutant):    
        A_c[ind[i], :] = np.random.rand(1, nb_params)
    
        # scale A between max and min value of objectives
        A_c[ind[i], 0] = min1+max1*A_c[ind[i], 0]
        A_c[ind[i], 1] = min2+max2*A_c[ind[i], 1]

    # merge parents and children
    A = np.concatenate((A_p, A_c), axis=0)
    
    # evaluate objective
    for i in range(pop_size):
        Ao[i] = A[i, 0]*np.sin(A[i, 0])*A[i, 1]*np.cos(A[i, 1])

    # sort A and Ao by ascending objective value
    A_ind = np.argsort(Ao)
    Ao = Ao[A_ind]
    A = A[A_ind, :]

    # add best individual to best_list
    best_list.append(Ao[0])
    gen_list.append(q+1)

    # plot current population 
    ax1.cla()
    ax1.plot(A[:, 0], A[:, 1], 'ro')
    ax1.plot(A[0, 0], A[0, 1], 'ko', markersize=10)  # the best individual is the first of the list
    ax1.contour(X, Y, Z, levels=levels)
    ax1.set_xlabel("design parameter x")
    ax1.set_ylabel("design parameter y")
    ax1.set_title("design space")

    ax2.cla()
    ax2.plot(gen_list, best_list)
    ax2.set_xlabel("Number of generations")
    ax2.set_ylabel("Best objective value")
    ax2.set_title("objective space")

    filename = 'plots/frame{:02d}.png'.format(q)
    plt.savefig(filename, bbox_inches='tight')
    plt.pause(0.1)

best = A[0, :]
best_o = Ao[0]
print('Best indiviual parameters: ', best)
print('Best indiviual objective: ', best_o)

plt.ioff()    
# plt.show()

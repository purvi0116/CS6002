import argparse
import numpy as np
import pulp
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import matplotlib.pyplot as plt
from numpy import arange

MAX = 1000
epsilon = 1e-6

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--users",type=int, default=4)
parser.add_argument("-i", "--intervals", type=int, default=3)
parser.add_argument("-l", "--lengths",type=list, default=[1, 1, 1])
parser.add_argument("-p", "--peaks",type=list, default=[1, 1, 1, 3])
parser.add_argument("-a", "--active",type=list, default=[[0, 3], [1, 3], [2, 3]])
args = parser.parse_args()

users = args.users
num_intervals = args.intervals
peaks = args.peaks
interval_lengths = args.lengths
total_length = np.sum(interval_lengths)
active_agents = args.active

# create the literals
literals = []
for i in range(num_intervals):
    literals.append([])
    for j in range(users):
        literals[i].append(LpVariable(name=f"alloc_{i}_{j}", lowBound=0))

model = LpProblem(name="process", sense=LpMaximize)

# add the constraints
sum = 0
# peaks constraint
for i in range(users):
    alloted = 0
    for j in range(num_intervals):
        alloted += literals[j][i]
    model.addConstraint(alloted <= peaks[i])
    sum += alloted

# add the objective function
model.setObjective(sum)
   
# interval constraint
for i in range(num_intervals):
    model.addConstraint(lpSum(literals[i]) <= interval_lengths[i])

for i in range(num_intervals):
    for j in range(users):
        if j not in active_agents[i]:
            model.addConstraint(literals[i][j] == 0)

# solve the model
status = model.solve()

# print the results
print("Status:", LpStatus[status])

sum_allocated = model.objective.value()

print("Objective value:", sum_allocated)

# print variable values
for i in range(num_intervals):
    for idx, elem in enumerate(active_agents[i]):
        print(f"alloc_{i}_{elem} = {literals[i][elem].value()}")


# plot the results
fig, grph = plt.subplots()
grph.set_title("Time alloted to each agent")
grph.set_ylim(0, np.sum(interval_lengths)+1)
grph.set_xlim(0, users+1)
grph.set_xlabel('time required to complete the task (peaks)')
grph.set_ylabel('time')
grph.set_xticks(arange(1, users+1, 1))
grph.set_xticklabels(peaks)
grph.grid(True)

start = 0
for i, row in enumerate(literals):
    offset = 0
    for j, variable in enumerate(row):
        grph.broken_barh([(j+1, 0.2)], (start+offset, variable.value()), facecolors =('tab:green'))
        offset += variable.value()
    start += interval_lengths[i]

plt.savefig("process_linear_alloc_max.png")

############################################ Envyfreeness ############################################
literals2 = []
for i in range(num_intervals):
    literals2.append([])
    for j in range(users):
        literals2[i].append(LpVariable(name=f"alloc2_{i}_{j}", lowBound=0))

model2 = LpProblem(name="envyfree", sense=LpMaximize)

min_value = []

for i in range(users):
    m = []
    for j in range(users):
        m.append(LpVariable(name=f"min_{i}_{j}", lowBound=0))
    min_value.append(m)

model2.setObjective(lpSum([min_value[i][j] for i in range(users) for j in range(users)]))

sum2 = 0
for i in range(users):
    alloted = 0
    for j in range(num_intervals):
        alloted += literals2[j][i]
    sum2 += alloted
    model2.addConstraint(alloted <= peaks[i])

    active_i = []

    for j in range(len(active_agents)):
        if i in active_agents[j]:
            active_i.append(j)

    for j in range(users):
        alloted_others = 0
        for k in active_i:
            alloted_others += literals2[k][j]

        model2.addConstraint(min_value[i][j] <= alloted)
        model2.addConstraint(min_value[i][j] <= alloted_others)
        model2.addConstraint(min_value[i][j] <= peaks[i])

model2.addConstraint(sum2 == sum_allocated)

for i in range(num_intervals):
    model2.addConstraint(lpSum(literals2[i]) <= interval_lengths[i])

for i in range(num_intervals):
    for j in range(users):
        if j not in active_agents[i]:
            model2.addConstraint(literals2[i][j] == 0)

# mysolver = pulp.PULP_CHOCO_CMD()
status2 = model2.solve()

print("Status:", LpStatus[status2])
print("Objective value (SUM):", model2.objective.value())

# print variable values
for i in range(num_intervals):
    for idx, elem in enumerate(active_agents[i]):
        print(f"alloc2_{i}_{elem} = {literals2[i][elem].value()}")

for i in range(users):
    for j in range(users):
        print(f"min_{i}_{j} = {min_value[i][j].value()}")

fig, grph = plt.subplots()
grph.set_title("Time alloted to each agent")
grph.set_ylim(0, np.sum(interval_lengths)+1)
grph.set_xlim(0, users+1)
grph.set_xlabel('time required to complete the task (peaks)')
grph.set_ylabel('time')
grph.set_xticks(arange(1, users+1, 1))
grph.set_xticklabels(peaks)
grph.grid(True)

start = 0
for i, row in enumerate(literals2):
    offset = 0
    for j, variable in enumerate(row):
        grph.broken_barh([(j+1, 0.2)], (start+offset, variable.value()), facecolors =('tab:green'))
        offset += variable.value()
    start += interval_lengths[i]

plt.savefig("process_linear_envyfree.png")


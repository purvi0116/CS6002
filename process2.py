import numpy as np
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import matplotlib.pyplot as plt
from numpy import arange

M = 1000
num_agents = 4
num_intervals = 8
peaks = [2, 4, 3, 2]
# peaks = [2, 2, 3]
# peaks = [4, 4, 5]
interval_lengths = [1, 1, 1, 2, 1, 1, 1, 2]
# interval_lengths = [3, 5] 
# interval_lengths = [4, 6]
total_length = np.sum(interval_lengths)
connected_agents = [[], [0], [0, 1], [1], [1, 3], [3], [2, 3], [2]]
# connected_agents = [[0], [0, 1, 2]] 
# connected_agents = [[0, 1], [1, 2]]

# create the variables
variables = []
for i in range(num_intervals):
    variables.append([])
    for j in range(num_agents):
        variables[i].append(LpVariable(name=f"alloc_{i}_{j}", lowBound=0))

var_comp = []
for i in range(num_agents) :
    var_comp.append(LpVariable(name=f"alloc_{i}", cat="Binary"))

# create the model
model = LpProblem(name="process", sense=LpMaximize)

# add the objective function
model.setObjective(lpSum(var_comp))

# add the constraints

# peaks constraint
for i in range(num_agents):
    alloted = 0
    for j in range(num_intervals):
        alloted += variables[j][i]
    model.addConstraint(alloted <= peaks[i])
    
    # var_comp constraint
    # if alloted is equal to peaks[i] then var_comp[i] = 1
    # if alloted is less than peaks[i] then var_comp[i] = 0
    model.addConstraint(M*(var_comp[i] - 1) <= (alloted - peaks[i]))

# interval constraint
for i in range(num_intervals):
    model.addConstraint(lpSum(variables[i]) <= interval_lengths[i])

for i in range(num_intervals):
    for j in range(num_agents):
        if j not in connected_agents[i]:
            model.addConstraint(variables[i][j] == 0)

# solve the model
status = model.solve()

# print the results
print("Status:", LpStatus[status])

num_allocated = model.objective.value()
print("Objective value:", num_allocated)


# print variable values
for i in range(num_intervals):
    for idx, elem in enumerate(connected_agents[i]):
        print(f"alloc_{i}_{elem} = {variables[i][idx].value()}")

for i in range(num_agents):
    print(f"alloc_{i} = {var_comp[i].value()}")


# plot the results
fig, grph = plt.subplots()
grph.set_title("Time alloted to each agent")
grph.set_ylim(0, np.sum(interval_lengths)+1)
grph.set_xlim(0, num_agents+1)
grph.set_xlabel('time required to complete the task (peaks)')
grph.set_ylabel('time')
grph.set_xticks(arange(1, num_agents+1, 1))
grph.set_xticklabels(peaks)
grph.grid(True)

start = 0
for i, row in enumerate(variables):
    offset = 0
    for j, variable in enumerate(row):
        grph.broken_barh([(j+1, 0.2)], (start+offset, variable.value()), facecolors =('tab:green'))
        offset += variable.value()
    start += interval_lengths[i]

plt.savefig("max_process_alloc.png")

############################################ Minimizing off time ############################################
variables2 = []
for i in range(num_intervals):
    variables2.append([])
    for j in range(num_agents):
        variables2[i].append(LpVariable(name=f"alloc2_{i}_{j}", lowBound=0))

var_comp2 = []
for i in range(num_agents) :
    var_comp2.append(LpVariable(name=f"alloc2_{i}", cat="Binary"))


model2 = LpProblem(name="wastage", sense=LpMaximize)

sum = 0
for i in range(num_agents):
    alloted = 0
    for j in range(num_intervals):
        alloted += variables2[j][i]
    sum += alloted
    model2.addConstraint(alloted <= peaks[i])
    model2.addConstraint(peaks[i] >= alloted + M*(var_comp2[i]-1))
    model2.addConstraint(peaks[i] <= alloted + M*(var_comp2[i]))

model2.addConstraint(lpSum(var_comp2) == num_allocated)

model2.setObjective(sum)

for i in range(num_intervals):
    model2.addConstraint(lpSum(variables2[i]) <= interval_lengths[i])

for i in range(num_intervals):
    for j in range(num_agents):
        if j not in connected_agents[i]:
            model2.addConstraint(variables2[i][j] == 0)


status2 = model2.solve()

print("Status:", LpStatus[status2])
print("Objective value (SUM):", model2.objective.value())

# print variable values
for i in range(num_intervals):
    for idx, elem in enumerate(connected_agents[i]):
        print(f"alloc2_{i}_{elem} = {variables2[i][idx].value()}")

for i in range(num_agents):
    print(f"alloc2_{i} = {var_comp2[i].value()}")

# plot the results
fig, grph = plt.subplots()
grph.set_title("Time alloted to each agent")
grph.set_ylim(0, np.sum(interval_lengths)+1)
grph.set_xlim(0, num_agents+1)
grph.set_xlabel('time required to complete the task (peaks)')
grph.set_ylabel('time')
grph.set_xticks(arange(1, num_agents+1, 1))
grph.set_xticklabels(peaks)
grph.grid(True)

start = 0
for i, row in enumerate(variables2):
    offset = 0
    for j, variable in enumerate(row):
        grph.broken_barh([(j+1, 0.2)], (start+offset, variable.value()), facecolors =('tab:green'))
        offset += variable.value()
    start += interval_lengths[i]

plt.savefig("min_process_alloc_wastage.png")


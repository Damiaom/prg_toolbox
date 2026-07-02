import numpy as np
import copy
import numpy.random as rnd
import matplotlib.pyplot as plt
import gc
import matplotlib.cm as cm

def next_step(state, active_list, Lambda):
    L = len(state)
    index = rnd.randint(0, len(active_list))
    p = Lambda*1.0/(1+Lambda)

    random_site = active_list[index]
    y = random_site[0]
    x = random_site[1]

    assert state[y, x] == 1, 'An inactive site was selected. Something went wrong!'

    if rnd.rand() < p:
        nn = rnd.randint(1,5)
        if nn == 1: (y, x) = (y, (x+1)%L)
        if nn == 2: (y, x) = (y, (x-1)%L)
        if nn == 3: (y, x) = ((y+1)%L, x)
        if nn == 4: (y, x) = ((y-1)%L, x)

        if state[y,x] == 0:
            state[y,x] = 1
            active_list.append([y,x])

    else:
        state[y, x] = 0
        del active_list[index]

    return state, active_list

def single_seed(L):
    state = np.zeros((L,L), dtype = int)
    state[(L-1)/2,(L-1)/2] = 1
    return state

def random_state(L):
    state = rnd.randint(2, size = (L,L), dtype = int)
    return state

def init_steady_state(L, time_length, Lambda, random_init):
    count = 0
    timesteps = int(time_length + 2)

    while count < time_length or len(active_list) == 0:
        gc.collect()
        if random_init:
            state = random_state(L)
        else:
            state = single_seed(L)

        active_list = list(np.argwhere(state == 1))
        time = np.zeros(timesteps)

        for i in range(timesteps-1):
            state, active_list = next_step(state, active_list, Lambda)
            if len(active_list) == 0: break
            else: count += 1
    print('End init')
    return state

def find_avalanche(L, time_length, Lambda, lag, random_init = True):
    states_list = []
    density_matrix = np.zeros((L,L))

    timesteps = int(time_length + 2)

    steady_state = init_steady_state(L, time_length, Lambda, random_init)
    n_active_0 = np.count_nonzero(steady_state)
    active_list_0 = list(np.argwhere(steady_state == 1))

    while len(states_list) < time_length/lag or 0 in density_matrix or time_length+1 in density_matrix:
        gc.collect()
        states_list[:] = []

        state = copy.deepcopy(steady_state)
        time = np.zeros(timesteps)

        active_list = copy.deepcopy(active_list_0)

        for i in range(timesteps-1):
            time[i+1] = time[i] + 1/(1.0*len(active_list))
            if i%lag == 0:
                states_list.append(copy.deepcopy(state))
            next_step(state, active_list, Lambda)
            if len(active_list) == 0: break

        density_matrix = np.zeros((L,L))
        for i in range(len(states_list)):
            density_matrix += states_list[i]

    print('Found.')
    print('Printing image...')

    density_matrix_temp = np.ma.masked_where(density_matrix == 0, density_matrix)
    fig, ax = plt.subplots()
    cmap = copy.copy(cm.get_cmap("OrRd"))
    cmap.set_bad(color='white')
    c = ax.imshow(density_matrix_temp, cmap = cmap)
    fig.colorbar(c, ax=ax)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    time_lag = np.zeros(len(states_list))
    for i in range(len(time)-1):
        if i%lag == 0: time_lag[i//lag] = time[i]

    return states_list, time_lag

def linearize_lattice(states_list):
    x_timeseries = np.zeros((len(states_list), states_list[0].shape[0]**2))
    print('Unraveling neurons...')
    for i in range(x_timeseries.shape[0]):
        x_timeseries[i, :] = states_list[i].ravel()
    print('Done.')

    return x_timeseries

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# Global variables
NUM_TRAINING_EPOCHS = 50
NUM_DATAPOINTS_PER_EPOCH = 64
NUM_TRAJ_SAMPLES = 10
DELTA_T = 0.05
rng = np.random.RandomState(12345)

def sim_rollout(sim, policy, n_steps, dt, init_state):
    """
    :param sim: the simulator
    :param policy: policy that generates rollout
    :param n_steps: number of time steps to run
    :param dt: simulation step size
    :param init_state: initial state

    :return: times:   a numpy array of size [n_steps + 1]
             states:  a numpy array of size [n_steps + 1 x 4]
             actions: a numpy array of size [n_steps]
                        actions[i] is applied to states[i] to generate states[i+1]
    """
    states = []
    state = init_state
    actions = []
    for i in range(n_steps):
        states.append(state)
        action = policy.predict(state)
        actions.append(action)
        state = sim.step(state, [action], noisy=True)

    states.append(state)
    times = np.arange(n_steps + 1) * dt
    return times, np.array(states), np.array(actions)


def augmented_state(state, action):
    """
    :param state: cartpole state
    :param action: action applied to state
    :return: an augmented state for training GP dynamics
    """
    dtheta, dx, theta, x = state
    return x, dx, dtheta, np.sin(theta), np.cos(theta), action


def make_training_data(state_traj, action_traj, delta_state_traj):
    """
    A helper function to generate training data.
    """
    x = np.array([augmented_state(state, action) for state, action in zip(state_traj, action_traj)])
    y = delta_state_traj
    return x, y

if __name__ == '__main__':

    # fix from https://blog.csdn.net/whereismatrix/article/details/78195783
    import matplotlib #fix problem from matplotlib
    matplotlib.use('TkAgg') #fix problem from matplotlib
    from cartpole_sim import CartpoleSim
    from policy import SwingUpAndBalancePolicy, RandomPolicy

    swingup_policy = SwingUpAndBalancePolicy('policy.npz')
    random_policy = RandomPolicy(seed=12831)
    sim = CartpoleSim(dt=DELTA_T)

    # Initial training data used to train GP for the first epoch
    init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)
    ts, state_traj, action_traj = sim_rollout(sim, random_policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
    delta_state_traj = state_traj[1:] - state_traj[:-1]
    train_x, train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)

    # np.savez('my_training_data_0.npz', x = train_x, y = train_y)
    
    # test

    # print(repr(train_x))
    # print("\n")
    # print(repr(train_y))

    for epoch in range(NUM_TRAINING_EPOCHS):
        # vis.clear()

        # Use learned policy every 4th epoch
        if (epoch + 1) % 4 == 0:
            policy = swingup_policy
            init_state = np.array([0.01, 0.01, 0.05, 0.05]) * rng.randn(4)
        else:
            policy = random_policy
            init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)

        ts, state_traj, action_traj = sim_rollout(sim, policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        delta_state_traj = state_traj[1:] - state_traj[:-1]

        # Augment training data
        new_train_x, new_train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)
        train_x = np.concatenate([train_x, new_train_x])
        train_y = np.concatenate([train_y, new_train_y])

    np.savez('my_training_data_noGP_3ep.npz', x = train_x, y = train_y)

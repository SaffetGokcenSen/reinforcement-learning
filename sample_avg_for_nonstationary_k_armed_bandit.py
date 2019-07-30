# implements exercise 2.5 in the book named "Reinforcement Learning An Introduction" by Richard S. Sutton and Andrew G.
# Barto.
# The approximate action value function is updated incrementally as indicated  in the section 2.4 of the book.

# Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the difficulties that sample-average
# methods have for non-stationary problems. Use a modified version of the 10-armed testbed in which all the q(a) start
# out equal and then take independent random walks (say by adding a normally distributed increment with mean zero and
# standard deviation 0.01 to all the q(a) on each step). Prepare plots for an action-value method using sample averages,
# incrementally computed, and another action-value method using a constant step-size parameter, alpha = 0.1. Use
# epsilon = 0.1 and longer runs, say of 10,000 steps.

import numpy as np
import matplotlib.pyplot as plt

k = 10  # number of arms
m = 2000  # number of different k-bandit problems
the_mean = 0.0  # the mean of the normal distribution for the true value of the action value function q(t)
the_variance = 1.0  # the variance of the normal distribution for the true value of the action value function q(t)
num_of_timesteps = 10000  # number of time steps a bandit problem is run
method_array = ["sample_avg", "constant_step_size"]
color_array = ["black", "red"]
epsilon = 0.1  # the probability for epsilon greedy action
alpha = 0.1  # constant step size
t = range(num_of_timesteps)

for the_method, the_color in zip(method_array, color_array):
    # for each problem all of the q(a) start the same. Hence, there are only one mean value for each problem. All of the
    # q(a)'s will begin as a normal distribution with the common mean and variance 1.0
    testbed_vector = np.random.normal(the_mean, the_variance, m)

    reward_time_matrix = np.zeros((m, num_of_timesteps))

    for problem_index in range(m):
        mean_value = testbed_vector[problem_index]  # the common mean for the action value functions of a problem
        reward_dist_vector = np.random.normal(mean_value, 1.0, k)
        action_occurrence = np.zeros(k)
        estimated_q = np.zeros(k)
        reward_time_variation = np.zeros(num_of_timesteps)
        for time_step in range(num_of_timesteps):
            greedy = np.random.binomial(1, (1 - epsilon))
            if greedy == 1:
                max_estimated_q = np.max(estimated_q)
                indices_max_estimated_q = np.where(estimated_q == max_estimated_q)[0]
                if len(indices_max_estimated_q) == 1:
                    action_index = indices_max_estimated_q[0]
                else:
                    action_index = np.random.choice(indices_max_estimated_q)
            else:
                action_index = np.random.choice(k)

            reward = np.random.normal(reward_dist_vector[action_index], 1.0)
            reward_dist_vector = reward_dist_vector + np.random.normal(0.0, 0.01, k)
            if the_method == "sample_avg":
                action_occurrence[action_index] = action_occurrence[action_index] + 1
                estimated_q[action_index] = estimated_q[action_index] + (1 / action_occurrence[action_index]) * (reward - estimated_q[action_index])
            elif the_method == "constant_step_size":
                estimated_q[action_index] = estimated_q[action_index] + alpha * (reward - estimated_q[action_index])

            reward_time_variation[time_step] = reward
        reward_time_matrix[problem_index] = reward_time_variation

    average_reward_time = np.mean(reward_time_matrix, axis=0)

    plt.plot(t, average_reward_time, the_color, label=the_method)
    plt.legend()


plt.xlabel("time")
plt.ylabel("average reward")
plt.title("epsilon greedy with sample average method and \n the constant step size method for non-stationary problems")
plt.grid()
plt.savefig("test_incremental.png")
plt.show()

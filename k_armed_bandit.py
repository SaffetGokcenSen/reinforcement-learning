# implements the 10-armed testbed in 2.3 of the book named "Reinforcement Learning An Introduction" by Richard S. Sutton
# and Andrew G. Barto
import numpy as np
import matplotlib.pyplot as plt

k = 10  # number of arms
m = 2000  # number of different k-bandit problems
the_mean = 0.0  # the mean of the normal distribution for the true value of the action value function q(a)
the_variance = 1.0  # the variance of the normal distribution for the true value of the action value function q(a)
num_of_timesteps = 2000  # number of time steps a bandit problem is run
epsilon_array = [0.0, 0.01, 0.1]
color_array = ["black", "red", "blue"]
t = range(num_of_timesteps)

for epsilon, the_color in zip(epsilon_array, color_array):
    # each row has the true values of the k action value functions, there are m rows each of which corresponds to an
    # independent bandit problem
    testbed_matrix = np.random.normal(the_mean, the_variance, (m, k))

    reward_time_matrix = np.zeros((m, num_of_timesteps))

    for problem_index in range(m):
        mean_value = testbed_matrix[problem_index]
        total_reward = np.zeros(k)
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

            reward = np.random.normal(mean_value[action_index], 1.0)
            total_reward[action_index] = total_reward[action_index] + reward
            action_occurrence[action_index] = action_occurrence[action_index] + 1
            estimated_q[action_index] = total_reward[action_index] / action_occurrence[action_index]
            reward_time_variation[time_step] = reward
        reward_time_matrix[problem_index] = reward_time_variation

    average_reward_time = np.mean(reward_time_matrix, axis=0)

    plt.plot(t, average_reward_time, the_color, label=repr(epsilon))
    plt.legend()


plt.xlabel("time")
plt.ylabel("average reward")
plt.title("epsilon greedy with varying epsilon")
plt.grid()
plt.savefig("test.png")
plt.show()

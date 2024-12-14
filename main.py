# This is a sample Python script.

# Press Shift+f10 to execute it or replace it with your code.
# Press double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')  # set interface for interactive plots


def rejection_condition(node_values, condition):
    """
    Checks if C = c- or F = f- of G = g-
    :return:
    True if condition is broken, False otherwise
    """
    for c_node, c_value in condition.items():  # length(node_values) = length(condition)
        # check if for the current node there is a condition
        if c_value is None:
            continue
        else:
            # Check is the node has been generated
            if node_values[c_node] is None:
                continue
            # check if there are nor equal
            elif node_values[c_node] != condition[c_node]:
                return True
    return False


def generate_sample(condition):
    """
    Generates a sample based on the CPTs, throws away the sample is rejection_condition is triggered
    :param condition:
    This the condition by which the sample will be rejected
    :return:
    1 if the sample is what we are looking for 0 otherwise;
    """
    # Initiate sample and sample for a
    curr_node_values = {'A': np.random.rand() > 0.6,
                        'B': None,
                        'C': None,
                        'D': None,
                        'E': None,
                        'F': None,
                        'G': None
                        }

    # After each node generation we check if the condition has been broken
    if rejection_condition(curr_node_values, condition):
        return 0, curr_node_values

    # Sample node b based on a
    if curr_node_values['A']:
        curr_node_values['B'] = np.random.rand() > 0.2  # P(b+|a+) = 0.8
    else:
        curr_node_values['B'] = np.random.rand() > 0.7  # P(b+|a-) = 0.3

    # check if the condition has been broken
    if rejection_condition(curr_node_values, condition):
        return 0, curr_node_values

    # Sample node c based on a
    if curr_node_values['A']:
        curr_node_values['C'] = np.random.rand() > 0.3  # P(c+|a+) = 0.7
    else:
        curr_node_values['C'] = np.random.rand() > 0.7  # P(c+|a-) = 0.3

    # check if the condition has been broken
    if rejection_condition(curr_node_values, condition):
        return 0, curr_node_values

    # Sample node d based on b
    if curr_node_values['B']:
        curr_node_values['D'] = np.random.rand() > 0.2  # P(d+|b+) = 0.2
    else:
        curr_node_values['D'] = np.random.rand() > 0.6  # P(d+|b-) = 0.4

    # check if the condition has been broken
    if rejection_condition(curr_node_values, condition):
        return 0, curr_node_values

    # Sample node e based on b and c
    if curr_node_values['B'] and curr_node_values['C']:
        curr_node_values['E'] = np.random.rand() > 0.8  # P(e+|b+, c+) = 0.2
    elif curr_node_values['B'] and not curr_node_values['C']:
        curr_node_values['E'] = np.random.rand() > 0.2  # P(e+|b+, c-) = 0.8
    elif not curr_node_values['B'] and curr_node_values['C']:
        curr_node_values['E'] = np.random.rand() > 0.5  # P(e+|b-, c+) = 0.5
    else:
        curr_node_values['E'] = np.random.rand() > 0.1  # P(e+|b-, c-) = 0.9

    # check if the condition has been broken
    if rejection_condition(curr_node_values, condition):
        return 0, curr_node_values

    # Sample node f based on e
    if curr_node_values['E']:
        curr_node_values['F'] = np.random.rand() > 0.6  # P(f+|e+) = 0.4
    else:
        curr_node_values['F'] = np.random.rand() > 0.4  # P(f+|e-) = 0.6

    # check if the condition has been broken
    if rejection_condition(curr_node_values, condition):
        return 0, curr_node_values

    # Sample node g based on e
    if curr_node_values['E']:
        curr_node_values['G'] = np.random.rand() > 0.3  # P(g+|e+) = 0.7
    else:
        curr_node_values['G'] = np.random.rand() > 0.8  # P(g+|e-) = 0.2

    # check if the condition has been broken
    if rejection_condition(curr_node_values, condition):
        return 0, curr_node_values

    return 1, curr_node_values


def calculate_c_pos_f_pos_g_pos():
    """
    Calculates probability of the specified conditional statement
    p(c+|f+,g+)
    :return:
    #(*, *, c+, *, *, f+, g+) / #(*, *, *, *, *, f+, g+)
    """
    # Define the condition P(c+|f+, g+) # None means '*', it can be anything
    # set condition for the upper part of the equation
    conditions_c_pos = {
        'A': None,
        'B': None,
        'C': True,
        'D': None,
        'E': None,
        'F': True,
        'G': True
    }
    # set condition for the lower part of the equation
    conditions_c_star = {
        'A': None,
        'B': None,
        'C': None,
        'D': None,
        'E': None,
        'F': True,
        'G': True
    }

    # define initial parameters
    N = 10000  # number of iterations
    sample_c_pos = []  # to store the value for the upper
    sample_c_star = []  # to store the value for the upper
    for i in range(N):
        # generate a sample and determine if it favorable or not
        sample_c_pos.append(generate_sample(conditions_c_pos)[0])
        sample_c_star.append(generate_sample(conditions_c_star)[0])
    # p(c+|f+,g+) = #(*,*,c+,*,*,f+,g+) / #(*,*,*,*,*,f+,g+)
    c_pos_f_pos_g_pos = sum(sample_c_pos) / sum(sample_c_star)
    # return the finale result
    return c_pos_f_pos_g_pos


def plot_data(data):
    # Calculate the mean (mathematical expectation) and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)

    print(f"Mean (Mathematical Expectation): {mean}")
    print(f"Standard Deviation: {std_dev}")

    # Plot the histogram with the mean and standard deviation marked
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean, plt.ylim()[1] * 0.9, f'Mean: {mean}', color='red')

    # Mark one standard deviation on either side of the mean
    plt.axvline(mean - std_dev, color='green', linestyle='dashed', linewidth=1)
    plt.axvline(mean + std_dev, color='green', linestyle='dashed', linewidth=1)
    plt.text(mean - std_dev, plt.ylim()[1] * 0.85, f'-1 STD', color='green')
    plt.text(mean + std_dev, plt.ylim()[1] * 0.85, f'+1 STD', color='green')

    plt.title('Histogram of Data with Mean and STD Marked')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def task_1_b():
    """
    Shel function for task 1
    :return:
    Void
    """
    number_of_exp = 100  # number of times we repeat the experiment
    data = []  # place to
    # run Nr times the experiment
    for i in range(number_of_exp):
        # add to array of data
        data.append(calculate_c_pos_f_pos_g_pos())
    plot_data(data)


def initialize_states():
    """ Initialize random states for each node in the Bayesian network. """
    return {node: np.random.choice([0, 1]) for node in ['A', 'B', 'C', 'D', 'E', 'F', 'G']}


def norm_ret_neg_prob(list_neg, list_pos):
    """
    Calculates probability of the specified conditional statement and normalizes them
    :param list_pos:
    List of pos probabilities
    :param list_neg:
    List of neg probabilities
    :return:
    The probability of the node to be 0 ('minus') !!!
    """
    tmp_pos = 1
    for i in range(len(list_pos)):
        tmp_pos *= list_pos[i]
    tmp_neg = 1
    for i in range(len(list_neg)):
        tmp_neg *= list_neg[i]
    # Normalization factor alfa
    alfa = 1 / (tmp_neg + tmp_pos)
    return alfa * tmp_neg


def sample_conditional(node, given_states):
    """ Sample a new state for a node based on the conditional probabilities of the network. """
    if node == 'A':  # p(A|B,C,D,E,F,G) = p(A|B,C) = p(B|A)*p(C|A)*p(A)
        # Calculate probabilities for A-
        p_b_given_a_neg = 0.7 if given_states['B'] == 0 else 0.3
        p_c_given_a_neg = 0.7 if given_states['C'] == 0 else 0.3
        p_a_neg = 0.6
        prob_list_neg = [p_b_given_a_neg, p_c_given_a_neg, p_a_neg]

        # Calculate probabilities for A+
        p_b_given_a_pos = 0.2 if given_states['B'] == 0 else 0.8
        p_c_given_a_pos = 0.3 if given_states['C'] == 0 else 0.7
        p_a_pos = 1 - p_a_neg
        prob_list_pos = [p_b_given_a_pos, p_c_given_a_pos, p_a_pos]

        # calculate final probability distribution for 'A'
        p = norm_ret_neg_prob(prob_list_neg, prob_list_pos)

        return np.random.choice([0, 1], p=[p, 1-p])
    elif node == 'B':  # p(B|A, C, D, E, F, G) = alfa * p(D|B) * p(E|B,C) * p(B|A)
        # Calculate probabilities for B-
        p_d_given_b_neg = 0.4 if given_states['D'] == 0 else 0.6
        c, e = given_states['C'], given_states['E']
        key = f'{c}{e}'
        probs = {'00': 0.1, '01': 0.9, '10': 0.5, '11': 0.5}
        p_e_given_c_b_neg = probs[key]
        p_b_given_a_neg = 0.7 if given_states['A'] == 0 else 0.2
        prob_list_neg = [p_d_given_b_neg, p_e_given_c_b_neg, p_b_given_a_neg]

        # Calculate probabilities for B+
        p_d_given_b_pos = 0.8 if given_states['D'] == 0 else 0.2
        c, e = given_states['C'], given_states['E']
        key = f'{c}{e}'
        probs = {'00': 0.2, '01': 0.8, '10': 0.8, '11': 0.2}
        p_e_given_c_b_pos = probs[key]
        p_b_given_a_pos = 0.3 if given_states['A'] == 0 else 0.8
        prob_list_pos = [p_d_given_b_pos, p_e_given_c_b_pos, p_b_given_a_pos]

        # fide the final probability distribution for 'B'
        p = norm_ret_neg_prob(prob_list_neg, prob_list_pos)
        return np.random.choice([0, 1], p=[p, 1 - p])
    elif node == 'C':  # p(C|A, B, D, E, F, G) = alfa * p(E|B, C) * p(C|A)
        # for c-
        p_c_given_a_neg = 0.7 if given_states['A'] == 0 else 0.3
        b, e = given_states['B'], given_states['E']
        key = f'{b}{e}'
        probs = {'00': 0.1, '01': 0.9, '10': 0.2, '11': 0.8}
        p_e_given_c_b_neg = probs[key]
        prob_list_neg = [p_c_given_a_neg, p_e_given_c_b_neg]

        # for c+
        p_c_given_a_pos = 0.3 if given_states['A'] == 0 else 0.7
        b, e = given_states['B'], given_states['E']
        key = f'{b}{e}'
        probs = {'00': 0.5, '01': 0.5, '10': 0.8, '11': 0.2}
        p_e_given_c_b_pos = probs[key]
        prob_list_pos = [p_c_given_a_pos, p_e_given_c_b_pos]

        # the usual
        p = norm_ret_neg_prob(prob_list_neg, prob_list_pos)
        return np.random.choice([0, 1], p=[p, 1 - p])
    elif node == 'D':  # p(D| A, B, C, E, F, G) = alfa * p(D|B)
        # for D-
        p_d_given_b_neg = 0.4 if given_states['B'] == 0 else 0.8
        prob_list_neg = [p_d_given_b_neg]

        # for D+
        p_d_given_b_pos = 0.6 if given_states['B'] == 0 else 0.2
        prob_list_pos = [p_d_given_b_pos]

        p = norm_ret_neg_prob(prob_list_neg, prob_list_pos)
        return np.random.choice([0, 1], p=[p, 1 - p])
    elif node == 'E':
        # for E-
        b, c = given_states['B'], given_states['C']
        key = f'{b}{c}'
        probs = {'00': 0.1, '01': 0.5, '10': 0.2, '11': 0.8}
        p_e_given_c_b_neg = probs[key]
        p_f_given_e_neg = 0.6
        p_g_given_e_neg = 0.2
        prob_list_neg = [p_e_given_c_b_neg, p_f_given_e_neg, p_g_given_e_neg]

        # for E+
        b, c = given_states['B'], given_states['C']
        key = f'{b}{c}'
        probs = {'00': 0.9, '01': 0.5, '10': 0.8, '11': 0.2}
        p_e_given_c_b_pos = probs[key]
        p_f_given_e_pos = 0.4
        p_g_given_e_pos = 0.7
        prob_list_pos = [p_e_given_c_b_pos, p_f_given_e_pos, p_g_given_e_pos]

        p = norm_ret_neg_prob(prob_list_neg, prob_list_pos)
        return np.random.choice([0, 1], p=[p, 1 - p])


def gibbs_sampling(num_iterations, burn_in):
    """ Perform Gibbs sampling to estimate P(C+ | F+, G+). """
    # Initialize states
    cur_state = initialize_states()  # random state
    cur_state['F'], cur_state['G'] = 1, 1  # Conditioning on F+ and G+
    samples_c = []
    non_prof_nodes = ['A', 'B', 'C', 'D', 'E']

    for i in range(num_iterations):
        for node in non_prof_nodes:
            cur_state[node] = sample_conditional(node, cur_state)
        if i >= burn_in:
            samples_c.append(cur_state['C'])

    # Estimate the probability P(C+ | F+, G+)
    p_c_given_f_g = np.mean(samples_c)
    return p_c_given_f_g


def task_1_d():
    """
    Shel function for task 2
    :return:
    Void
    """
    num_iterations = 10000
    burn_in = num_iterations * 0.1  # 10% of all iterations
    number_of_exp = 100
    data = []
    for i in range(number_of_exp):
        # add to array of data
        data.append(gibbs_sampling(num_iterations, burn_in))
    plot_data(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Calling shel function for task 1, comment it if task 1 is not of interests
    # task_1_b()  # generating with throw away
    task_1_d()  # gibbs sampling
    print('Hello :)')
# See Pycharm help at https://www.jetbrains.com/help/pycharm/

# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import time

def evaluate_policy(env, gamma, policy, value_func, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.
    给出一个env和policy，计算各个状态的value以及value function收敛的迭代次数

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    value_func: np.array
      The value function array
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
      返回值：
        np.ndarray: numpy数组，存储每个状态对应的value值。也就是state-value function矩阵
        int: value function收敛所用的迭代次数
    """
    iterations = 0
    v = value_func

    for i in range(max_iterations):
        iterations += 1
        delta = 0
        v_old = v.copy()

        # iterate each state
        for s in range(env.nS):  # from 0 to 15 overall 16 states
            # evaluate policy 就是按照当前policy 给出的动作去走。 Note: 和value iteration 不一样，value iteration 里面先对s循环，在对a循环，说明对应一个s a不确定
            a = policy[s]  # 老师PPT里提到了，为了简化实现，这里每个转态下就只有唯一个动作。复杂的情况其实是一个distribution, 有多种可能
            expection = 0.0

            # ‘P’: Dynamics，是一个二维字典,第一维度是状态，第二维度是动作，值是对应状态和动作的能到达的下一个状态的四个属性 (概率, 下一个状态, 奖励, 是否终结状态)，
            #  可以到达多个状态, 即：P[s][a] = [(prob1, nextstate1, reward1, is_terminal1), (prob2, nextstate2, reward2, is_terminal2)]]
            # e.g. env.P[0][deeprl_p1.lake_envs.LEFT]表示在状态0，选择往左走, 我们得到返回值：[(1.0, 0, 0.0, False)]，
            # 如果是Deterministic，数组里只有一组元素，说明对应的下一个状态，到达这个状态的概率是100%。这个状态是0，奖励R(0, LEFT) = 0，不是终结状态。
            # ??? 转移概率矩阵 P[s][a] 什么时候初始化的 ？
            # ??? 为什么 is_terminal == True, 这么计算：expected_value +=  prob * (reward + gamma * 0)
            for prob, nextstate, reward, is_terminal in env.P[s][a]:
                # 根据value function 计算公式，要把可能走的方向的值都累加起来.就是期望
                if is_terminal == True:
                    expection += prob * (reward + gamma * 0)
                else:
                    expection += prob * (reward + gamma * v_old[nextstate]) # Note: here still use existing/old value_function value

            # update state value function
            v[s] = expection # update value function for the current state
            delta = max(delta, abs(v[s] - v_old[s]))

        # outside of the s for loop
        if (delta < tol):
            break

    return v, iterations


# deprecated, 项目中没有用到
def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    policy = np.zeros(env.nS, dtype='int')
    for s in range(env.nS):
        max_value = None
        # if take this action, calculate the expected reward
        for a in range(env.nA):
            expected_value = 0.0
            for prob, nextstate, reward, is_terminal in env.P[s][a]:
                if is_terminal:
                    # ??? 为什么在Policy Improvement 伪代码里，没有乘以prob?
                    expected_value +=  prob * (reward + gamma * 0)
                else:
                    expected_value +=  prob * (reward + gamma * value_function[nextstate])
            # Record the maximum value and corresponding action
            if max_value is None or max_value < expected_value:
                max_value = expected_value
                policy[s] = a

    return policy

# polic: Maps states to actions.
# 传入待提高的policy
def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.
        给出一个env，policy以及对应value_func，返回新的policy

        policy：策略, np.array, Maps states to actions, policy[s]=a

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """

    policy_stable = True
    for s in range(env.nS):
        old_action = policy[s] # note down the old action so that we can check whether policy get changed/udpated at the end of this function
        max_value = None

        for a in range(env.nA):
            expection = 0.0 # current state take different actions, then get the max value
            for prob, nextstate, reward, is_terminal in env.P[s][a]:
                if is_terminal:
                    expection +=  prob * (reward + gamma * 0)
                else:
                    expection += prob * (reward + gamma * value_func[nextstate]) # 这里不是更新value function,所以就用当前传入进来的的value_function value

            if max_value is None or max_value < expection:
                max_value = expection
                policy[s] = a # update policy here i.e. improve policy. Since we find larger Q(s, a) value

        if policy[s] != old_action: # current state's policy get changed -> policy is not stable yet
            policy_stable = False

    return policy_stable, policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """

    # 根据伪代码，初始化：state-value function, state policy 两个矩阵
    #   机器人如何根据状态选择一个动作? => 策略(Policy). 初始化： 方格里面的值 也就是每个state下，的policy
    #   机器人如何衡量一个状态的好坏?=>数值(Value function). 初始化： 方格里面的值 也就是每个state的value function 都是0
    policy = np.zeros(env.nS, dtype='int') # nS: state number i.e. 16个 （0， 1， 2... 14, 15）
    value_func = np.zeros(env.nS)

    improve_iteration = 0
    evalue_iteration = 0
    policy_stable = False

    for i in range(max_iterations):
        # tol: tolerate容忍值， 最大变化值小于tol被定义为收敛.
        # 传入policy, 评估一下这个policy, 得到一个更新后的value_func
        # e_iter： value function收敛所用的迭代次数
        value_func, e_iter = evaluate_policy(env, gamma, policy, value_func, max_iterations, tol)

        # 把评估好的policy, 已经evalut_policy产生的新的state-value function传入
        # return: 新的policy是否稳定了， 新的policy
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)

        improve_iteration += 1
        evalue_iteration += e_iter
        if policy_stable:
            break # stable了就退出循环，否则就继续循环，也就是进入另外一个 evaluate_policy， improve_policy 的cycle
    return policy, value_func, improve_iteration, evalue_iteration


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
        给出一个env，用value_iteration求出policy

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """

    # based on the 伪代码， randomly intialize below two matrix
    V = np.zeros(env.nS) # each env's state has a corresponding value (i.e. value function)
    policy = np.zeros(env.nS, dtype='int') # each env's state has a corresponiding policy (i.e. moving action)

    iteration_cnt = 0 # 控制iteration次数

    for i in range(max_iterations):
        delta = 0 # 记录下每一轮 (V_old[s]-V[s]) 的最大值，当这个最大值delta < tol的值, 就认为value function 收敛了, Q(s,a)也就是最优的了
        V_old = V.copy()  # based on the 伪代码， copy 当前的value function

        # 下面就是实现PPT上的伪代码， 下面两轮循环对应伪代码里 maxQ(s,a)
        for s in range(env.nS): # for each state
            max_value = None
            for a in range(env.nA):  # for each state, try all possible actions
                expectation = 0 # reset for each action
                # 当选择一个动作后， 迭代所有可能到达的的下一个状态. 由于这里选择了Deterministic模型(from example.py 的main function)，虽然是for循环，但只执行一轮， prob 也就是1，nextstate 也就只有1个状态
                for prob, nextstate, reward, is_terminal in env.P[s][a]:
                    if is_terminal:
                        expectation += prob * (reward + gamma * 0)  #ppt伪代码当中使用了E符号 也就是期望， 展开后开之后就有prob了
                    else:
                        expectation += prob * (reward + gamma * V_old[nextstate]) # Note: old existing value function
                        # expectation +=  prob * reward + gamma * prob * V[nextstate] 这样表述更match PPT里的伪代码

                if max_value is None or max_value < expectation:
                    max_value = expectation
                    policy[s] = a  #顺便也跟新了policy


            V[s] = max_value # update value function to new value for this state
            # max求出所有state的value function变化的最大值， 如果这个最大值小于阈值的话 就说明value function已经收敛了, policy也是最优的
            delta = max(delta, abs(V_old[s] - V[s]))

        # outside of s for loop, in side max_iteration for loop
        iteration_cnt += 1
        if delta < tol:
            break

    V[env.nS-1] = 0 #手动把最终状态置为0
    return V, policy, iteration_cnt


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    return str_policy

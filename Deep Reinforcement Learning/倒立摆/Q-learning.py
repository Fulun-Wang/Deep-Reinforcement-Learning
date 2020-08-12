import gym
import numpy as np

env = gym.make('CartPole-v0')

max_number_of_steps = 200  # 每一场游戏的最高得分
# 获胜的条件是最近100场平均得分高于195
goal_average_steps = 195
num_consecutive_iterations = 100
num_episodes = 5000
# 只存储最近100场的得分（可以理解为是一个容量为100的栈）
last_time_steps = np.zeros(num_consecutive_iterations)

# q_table是一个256*2的二维数组
# 离散化后的状态共有4^4=256种可能的取值，每种状态对应一个行动
# q_table[s][a]就是当状态为s时做出行动a的有利程度评价值

q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))
# print(q_table) 256*2的二维数组   size(256,2)
# print(env.action_space.n)  # 2
# print(env.action_space)  # Discrete(2) 离散


# 分箱处理函数，把[clip_min, clip_max]区间平均分为num段，位于i段区间的特征值x会被离散化为i
def bins(clip_min, clip_max, num):
    #  print(np.linspace(clip_min, clip_max, num + 1))
    return np.linspace(clip_min, clip_max, num + 1)[1:-1] # 从第一个取到最后一个
# np.linspace() 等差数列
# 离散化处理，将由4个连续特征值组成的状态矢量转换为一个0~~255的整数离散值
def digitize_state(observation):
    # 将矢量打散回4个连续特征值
    cart_pos, cart_v, pole_angle, pole_v = observation
    #  print(cart_v)
    #  print(cart_pos)
    #  print(np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)))
    #  分别对各个连续特征值进行离散化（分箱处理）
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]
    #  digitize在哪个区间
    #  print(digitized)
    #    for i, x in enumerate(digitized):
    #        print(i, x)
    #        print(x)

    #  print(sum([x * (4 ** i) for i, x in enumerate(digitized)]))
    #  将4个离散值再组合为一个离散值，作为最终结果
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])
    # enumerate 枚举


# 根据本次的行动及其反馈（下一个时间步的状态），返回下一次的最佳行动
def get_action(state, action, observation, reward, episode):
    next_state = digitize_state(observation)  # 获取下一个时间步的状态，并将其离散化
    epsilon = 0.5 * (0.99 ** episode)  # ع-贪心策略
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])

    alpha = 0.2  # 学习系数
    gamma = 0.99  # 报酬衰减系数
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                reward + gamma * q_table[next_state, next_action])
    # -------------------------------------------------------------------------------------------
    return next_action, next_state


for episode in range(num_episodes):
    observation = env.reset()  # 初始化本场游戏的环境
    state = digitize_state(observation)
    # del digitize_state(observation) 获取初始状态值  是一个sum
    #  print(state)
    #  print(q_table)
    #  print(q_table[state])
    action = np.argmax(q_table[state])  # 根据状态值做出行动决策
    #  print(action)
    episode_reward = 0

    # 一场游戏分为一个个时间步
    for t in range(max_number_of_steps):
        env.render()  # 更新并渲染游戏画面
        observation, reward, done, info = env.step(action)  # 获取本次行动的反馈结果
        # 对致命错误行动进行极大力度的惩罚,让模型狠狠地吸取教训
        # print(env.step(action))
        if done:
            print('kill!!!')
            reward = -200

        action, state = get_action(state, action, observation, reward, episode)  # del get_action
        episode_reward += reward
        if done:
            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))
            break

        if (last_time_steps.mean() >= goal_average_steps):
            print('Episode %d train agent successfuly!' % episode)
            break

print('Failed!')
env.close()
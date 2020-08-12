
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

lr = 0.01
epsilon = 0.9
gamma = 0.9
target_replace_iter = 100
i_state = 32 * 32  # 暂定
o_action = 4  # 下，右，左45度,右45度
memory_capacity = 1000
batch_size = 500


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print("x.shape is", x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # print("xx.shape is", x.shape)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        # 使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征
        # 即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value

    def num_flat_features(self, x):
        size = x.size()[1:]
        # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片
        # 那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class DDQN(object):
    def __init__(self):
        self.evaluate_net = Net()
        self.target_net = Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, i_state * 2 + 2))
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=lr)  # torch的优化器
        self.loss_func = nn.MSELoss()  # 均方误差 （x-y）**2

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)  # 将state(33,33)二维数组转化为三维的Tensor类型的数组
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 继续转化为四维Tensor数组
        # print(x.shape)
        if np.random.uniform() < epsilon:
            actions_value = self.evaluate_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            # print(action)
            # action_value最大值在数组中的位置，返回值是一个1*1的数组
        else:
            action = np.random.randint(0, o_action)
        return action

    def store_transiton(self, state, action, reward, next_state):
        state = state.flatten()
        # print(state.shape)
        next_state = state.flatten()
        # print(next_state.shape)
        transition = np.hstack((state, action, reward, next_state))
        # print(transition.shape)
        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.learn_step_counter += 1
        # print(memory_capacity)
        sample_index = np.random.choice(memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]

        b_state = b_memory[:, :i_state]  # (32, 1024)
        b_state = b_state.reshape(500, 32, 32)
        b_state = torch.unsqueeze(torch.FloatTensor(b_state), 1)
        # 上面三行将b_state变成一个Tensor类型的（32,1,32,32）的数组
        # b_state = torch.FloatTensor(b_memory[:, :i_state])

        b_action = torch.LongTensor(b_memory[:, i_state:i_state + 1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, i_state + 1:i_state + 2])

        b_next_state = b_memory[:, -i_state:]  # (32, 1024)
        b_next_state = b_next_state.reshape(500, 32, 32)
        b_next_state = torch.unsqueeze(torch.FloatTensor(b_next_state), 1)
        # 上面三行将b_next_state变成一个Tensor类型的（32,1,32,32）的数组
        # b_next_state = torch.FloatTensor(b_memory[:, -i_state:])

        q_evaluate = self.evaluate_net(b_state).gather(1, b_action)
        q_next_action = self.target_net(b_next_state).detach()
        q_target = b_reward + gamma * q_next_action.max(1)[0]
        loss = self.loss_func(q_evaluate, q_target)
        # print("loss is ", loss)  #  例如tensor(41819.9023, grad_fn=<MseLossBackward>)
        self.optimizer.zero_grad()  # 清空过往梯度
        loss.backward()  # 反向传播，计算当前梯度
        self.optimizer.step()  # 根据梯度更新网络参数


dqn = DDQN()
for i_episode in range(10000000):
    # 训练1000000次
    map = np.full((400, 400), int(10), dtype=np.int8)
    map[23, 25] = 2
    map[372, 367] = 5
    map[0:16, 0:400] = 0
    map[384:400, 0:400] = 0
    map[0:400, 0:16] = 0
    map[0:400, 384:400] = 0
    map[60:70, 15:64] = 0
    map[10:35, 46:54] = 0
    map[42:54, 78:86] = 0
    # map[18:27, 80:120] = 0
    # map[53:116, 110:116] = 0
    # map[78:169, 75:92] = 0
    map[147:154, 100:167] = 0
    map[87:95, 112:134] = 0
    # map[110:118, 133:186] = 0
    map[137:187, 110:119] = 0
    # map[200:249, 110:119] = 0
    # map[52:259, 201:210] = 0
    # map[187:249, 139:179] = 0
    map[277:283, 21:239] = 0
    map[250:299, 270:275] = 0
    # map[189:229, 257:305] = 0
    # map[157:162, 219:309] = 0
    map[320:369, 65:119] = 0
    # map[330:358, 139:259] = 0
    map[346:355, 256:323] = 0
    # map[220:349, 343:346] = 0
    # map[287:289, 270:326] = 0
    map[257:259, 318:376] = 0
    map[277:330, 167:172] = 0
    step = 0
    reward = 0
    x_position = 23
    y_position = 25
    distance = math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2)) / 100
    # s = math.sqrt(math.pow(372 - 23, 2) + math.pow(367 - 25, 2))
    # print(s)
    map[23, 25] = distance
    # print(distance)
    # print(map[23, 25])
    init_state = map[7:39, 9:41]
    state = init_state
    i_episode += 1
    print("第%d轮学习" % i_episode)
    # x = []
    # y = []
    while True:
        plt.ion()
        state_ = state  # 保存当前的状态
        action = dqn.choose_action(state)
        step += 1
        print("运动了%d步" % step)
        distance = math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2)) / 100
        if action == 0:
            next_position = np.array([x_position + 1, y_position + 1])
            x_position = next_position[0]
            y_position = next_position[1]
            next_distance = math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2)) / 100
            if map[x_position, y_position] != 0:

                if map[x_position, y_position] == 5:
                    print("成功到达目的地！！行动了%d步" % step)
                    break
                else:
                    map[x_position, y_position] = 4
                    plt.clf()
                    plt.imshow(map, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
                    x_ticks = np.arange(0, 400, 20)
                    y_ticks = np.arange(0, 400, 20)
                    plt.xlabel('y')
                    plt.ylabel('x')
                    plt.xticks(x_ticks, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                         '14', '15', '16', '17', '18', '19', '20'), color='blue')
                    plt.yticks(y_ticks, ('20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8',
                                         '7', '6', '5', '4', '3', '2', '1'), color='blue')
                    plt.grid(0)
                    plt.pause(0.001)
                    plt.show()
                    map[x_position, y_position] = distance
                    next_state = map[x_position - 16:x_position + 16, y_position - 16:y_position + 16]
                    if distance >= next_distance:
                        r = 10000 / math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2))
                        reward += r
                        print("东南，累计奖励%d" % reward)
                    else:
                        reward -= 1000
                        print("东南，累计奖励%d" % reward)
            else:
                reward -= 1000
                break
        elif action == 1:
            next_position = np.array([x_position - 1, y_position + 1])
            x_position = next_position[0]
            y_position = next_position[1]
            next_distance = math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2)) / 100
            if map[x_position, y_position] != 0:

                if map[x_position, y_position] == 5:
                    print("成功到达目的地！！行动了%d步" % step)
                    break
                else:
                    map[x_position, y_position] = 4
                    plt.clf()
                    plt.imshow(map, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
                    x_ticks = np.arange(0, 400, 20)
                    y_ticks = np.arange(0, 400, 20)
                    plt.xlabel('y')
                    plt.ylabel('x')
                    plt.xticks(x_ticks, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                         '14', '15', '16', '17', '18', '19', '20'), color='blue')
                    plt.yticks(y_ticks, ('20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8',
                                         '7', '6', '5', '4', '3', '2', '1'), color='blue')
                    plt.grid(0)
                    plt.pause(0.001)
                    plt.show()
                    map[x_position, y_position] = distance
                    next_state = map[x_position - 16:x_position + 16, y_position - 16:y_position + 16]
                    if distance >= next_distance:
                        r = 10000 / math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2))
                        reward += r
                        print("东南，累计奖励%d" % reward)
                    else:
                        reward -= 1000
                        print("东南，累计奖励%d" % reward)
            else:
                reward -= 1000
                break
        elif action == 2:
            next_position = np.array([x_position, y_position + 1])
            x_position = next_position[0]
            y_position = next_position[1]
            next_distance = math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2)) / 100
            if map[x_position, y_position] != 0:
                if map[x_position, y_position] == 5:
                    print("成功到达目的地！！行动了%d步" % step)
                    break
                else:
                    map[x_position, y_position] = 4
                    plt.clf()
                    plt.imshow(map, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
                    x_ticks = np.arange(0, 400, 20)
                    y_ticks = np.arange(0, 400, 20)
                    plt.xlabel('y')
                    plt.ylabel('x')
                    plt.xticks(x_ticks, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                         '14', '15', '16', '17', '18', '19', '20'), color='blue')
                    plt.yticks(y_ticks, ('20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8',
                                         '7', '6', '5', '4', '3', '2', '1'), color='blue')
                    plt.grid(0)
                    plt.pause(0.001)
                    plt.show()
                    map[x_position, y_position] = distance
                    next_state = map[x_position - 16:x_position + 16, y_position - 16:y_position + 16]
                    if distance >= next_distance:
                        r = 10000 / math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2))
                        reward += r
                        print("东南，累计奖励%d" % reward)
                    else:
                        reward -= 1000
                        print("东南，累计奖励%d" % reward)
            else:
                reward -= 1000
                break
        else:
            next_position = np.array([x_position + 1, y_position])
            x_position = next_position[0]
            y_position = next_position[1]
            next_distance = math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2)) / 100
            if map[x_position, y_position] != 0:
                if map[x_position, y_position] == 5:
                    print("成功到达目的地！！行动了%d步" % step)
                    break
                else:
                    map[x_position, y_position] = 4
                    plt.clf()
                    plt.imshow(map, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
                    x_ticks = np.arange(0, 400, 20)
                    y_ticks = np.arange(0, 400, 20)
                    plt.xlabel('y')
                    plt.ylabel('x')
                    plt.xticks(x_ticks, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                         '14', '15', '16', '17', '18', '19', '20'), color='blue')
                    plt.yticks(y_ticks, ('20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8',
                                         '7', '6', '5', '4', '3', '2', '1'), color='blue')
                    plt.grid(0)
                    plt.pause(0.001)
                    plt.show()
                    map[x_position, y_position] = distance
                    next_state = map[x_position - 16:x_position + 16, y_position - 16:y_position + 16]
                    if distance >= next_distance:
                        r = 10000 / math.sqrt(math.pow(372 - x_position, 2) + math.pow(367 - y_position, 2))
                        reward += r
                        print("东南，累计奖励%d" % reward)
                    else:
                        reward -= 1000
                        print("东南，累计奖励%d" % reward)
            else:
                reward -= 1000
                break
        state = next_state
        dqn.store_transiton(state_, action, reward, state)
        if dqn.memory_counter > memory_capacity:
            dqn.learn()

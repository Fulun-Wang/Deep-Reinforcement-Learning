"""
1.随机初始化Q网络和目标Q网络
2.以ϵ的概率随机选取动作action
3.执行动作a获取奖励reward和下一个状态 s′,更新s=s′
4.储存（s,a,r,s′）到经验池D中.
5.在经验池D中选取一组样本集（minibatch）,计算y=r+γmaxa′​target_Q(s′,a,w)
6.计算损失函数loss_function
7.使用梯度反向传播更新Q网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

batch_size = 32
LR = 0.01
epsilon = 0.9
gamma = 0.9
target_replace_iter = 100  # Q显示网络更新频率
memory_capacity = 2000  # 记忆库大小
env = gym.make('CartPole-v0')  # 使用gym中的Cartpole 环境
env = env.unwrapped  # 还原env的原始设置，env外包了一层放作弊层
n_action = env.action_space.n  # 2, 杆子能做的动作
n_states = env.observation_space.shape[0]  # 4,杆子能获取的环境信息数
env_a_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


# action = env.action_space.sample()   随机从动作空间中选取动作 -1 或 1
# print(env.action_space)  # 输出动作信息,Discrete(2)
# print(env.action_space.shape)  # ()
# print(env.action_space.n)  # 2,输出动作个数，动作：2个动作：施加-1和+1分别对应向左向右推动运载体
# print(env.observation_space)  # 查看状态空间，Box(4,)
# print(env.observation_space.shape)  # (4,)
# print(env.observation_space.shape[0]) # 4,输出状态的个数，状态：4个，x：位置;x_dot：移动速度, theta：角度 theta_dot：移动角速度
# print(env.observation_space.high)  # 查看状态的最高值
# print(env.observation_space.low)  # 查看状态的最低值


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 50)  # 4,50是指维度,输入样本大小为4，输出样本大小为10
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, n_action)  # 50,2是维度，输入样本大小为50，输出样本大小为2
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):
    def __init__(self):
        # 建立目标网络(target net)和价值网络(evaluate net)以及记忆库(memory)
        self.eval_net = Net()
        # print(self.eval_net)
        self.target_net = Net()
        self.learn_step_counter = 0  # target更新计时
        self.memory_counter = 0  # 记忆库计数
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        # print(type(n_states * 2 + 2))
        # 记忆库初始化 （2000,10） 注:10 = 4个state + 选择的1个action + 1个reward +  4个next state
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, state):
        # 根据环境观测observation选择动作
        x = torch.unsqueeze(torch.FloatTensor(state), 0)  # unsqueeze 对维度进行扩充
        # print(x.shape)
        if np.random.uniform() < epsilon:  # 随机产生一个【0,1】之间的数与epsilon比较 贪婪法
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()  # action_value最大值在数组中的位置，返回值是一个1*1的数组。
            # print(action)
            action = action[0] if env_a_shape == 0 else action.reshape(env_a_shape)  # reshape 重组数列
            # print(action)
            # output = torch.max(input, dim)
            # input是softmax函数输出的一个tensor
            # dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            # 在有的地方我们会看到torch.max(a, 1).data.numpy()的写法，
            # 这是因为在早期的pytorch的版本中，variable变量和tenosr是不一样的数据格式
            # variable可以进行反向传播，tensor不可以，需要将variable转变成tensor再转变成numpy
            # 现在的版本已经将variable和tenosr合并，所以只用torch.max(a,1).numpy()就可以了

        else:  # 选取随机动作
            action = np.random.randint(0, n_action)
            action = action if env_a_shape == 0 else action.reshape(env_a_shape)
        return action

    def store_transition(self, s, a, r, s_):
        # 存储记忆转移
        transition = np.hstack((s, a, r, s_))  # np.hstack():在水平方向上平铺
        # print(transition.shape)
        index = self.memory_counter % memory_capacity
        # 如果记忆库满了就覆盖老数据
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 目标网络（target net）更新
        if self.learn_step_counter % target_replace_iter == 0:  # 100次一更新
            self.target_net.load_state_dict(self.eval_net.state_dict())  # ？？？？
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(memory_capacity, batch_size)  # 随机从2000中选取32个
        # print(sample_index)  shape is [32,]
        b_memory = self.memory[sample_index, :]  # 存储【32,8】的数据
        # print(b_memory)
        b_s = torch.FloatTensor(b_memory[:, :n_states])  # 第一维全取，第二维取到n_states=4(取第0,1,2,3列)
        # print(b_s) shape is [32,4]
        b_a = torch.LongTensor(b_memory[:, n_states:n_states + 1].astype(int))  # 取第4列，astype:转换数据类型
        # print(b_a) # shape is [32,1]
        b_r = torch.FloatTensor(b_memory[:, n_states + 1:n_states + 2])  # 取第5列
        # print(b_r.shape)  # shape is [32,1]
        b_s_ = torch.FloatTensor(b_memory[:, -n_states:])  # 取最后四列
        # print(b_s) shape is [32,4]
        # 针对做过的动作b_a,选择q_eval 的值（q_eval 原本有所有动作的值）
        q_eval = self.eval_net(b_s).gather(1, b_a)  # dim=1(列索引) shape(batch=32,1)
        # print(self.eval_net(b_s))  # shape is torch.size([32,2])
        # print(q_eval.shape)  # shape is torch.size([32,1])
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        # print('q_next is:',q_next) # shape is [32,2]
        q_target = b_r + gamma * q_next.max(1)[0]  # shape is (batch=32, 32)  gamma is reward 衰减系数
        # torch.max(a,1)返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）,返回的最大值和索引各是一个tensor，一起构成元组(Tensor, LongTensor)
        loss = self.loss_func(q_eval, q_target)
        # 计算，更新eval_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

for i_episode in range(400):
    state = env.reset()  # 初始化本场环境
    # print(state.shape)
    while True:
        env.render()  # 更新并渲染游戏画面
        action = dqn.choose_action(state)  # 根据状态选取动作,从当前Q网络（q_eveal）选取的。
        # print(action.shape)
        next_state, r, done, info = env.step(action)  # 由选取的动作返回奖励和环境反馈
        x, x_dot, theta, theta_hot = next_state

        # 计算奖励reward
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(state, action, r, next_state)  # 存记忆

        if dqn.memory_counter > memory_capacity:  # 当记忆库满2000开始学习
            dqn.learn()

        if done:
            break
        s = next_state  # 更新环境

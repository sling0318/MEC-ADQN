import numpy as np
import torch
import torch.nn as nn
import torch.optim


a0 = np.full((256, 1), 0.)
a1 = np.full((256, 1), 1.)
a2 = np.full((256, 1), 2.)
a3 = np.full((256, 1), 3.)
a4 = np.full((256, 1), 4.)
a5 = np.full((256, 1), 5.)
a6 = np.full((256, 1), 6.)
a7 = np.full((256, 1), 7.)
a8 = np.full((256, 1), 8.)
a9 = np.full((256, 1), 9.)


a = np.vstack((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))

b0 = np.linspace(0, 1, 256).reshape((256, 1))
b1 = np.linspace(0, 1, 256).reshape((256, 1))
b2 = np.linspace(0, 1, 256).reshape((256, 1))
b3 = np.linspace(0, 1, 256).reshape((256, 1))
b4 = np.linspace(0, 1, 256).reshape((256, 1))
b5 = np.linspace(0, 1, 256).reshape((256, 1))
b6 = np.linspace(0, 1, 256).reshape((256, 1))
b7 = np.linspace(0, 1, 256).reshape((256, 1))
b8 = np.linspace(0, 1, 256).reshape((256, 1))
b9 = np.linspace(0, 1, 256).reshape((256, 1))

b = np.vstack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))

actions = np.hstack((a, b))

class Network(nn.Module):

    def __init__(self, n_features, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 50)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc1.bias.data.normal_(0.1)
        self.relu = nn.ReLU()
        self.out = nn.Linear(50, n_actions)
        self.out.weight.data.normal_(0, 0.3)
        self.out.bias.data.normal_(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.out(x)


class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=50,
                 e_greedy_increment=0.001,
                 output_graph=True,
                 epoch=100,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gama = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epoch = epoch

        self.epsilon = 0
        self.learn_step_counter = 0

        # 初始化replay action为2维参数
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
        self.memory_counter = 0
        # print(self.memory.shape)

        self.cost_his = []

        self.eval_net, self.target_net = Network(self.n_features, self.n_actions), Network(self.n_features, self.n_actions)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_fun = nn.MSELoss()


    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, r, s_))
        # print("需要存入的记录为：", transition)

        # 如果满了，替换调旧的transition
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        # print("memory:", self.memory)

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation[np.newaxis, :])    # 增加一个维度 i.e[1,2,3,4,5]变成[[1,2,3,4,5]]
        if np.random.uniform() < self.epsilon:
            # 选择q值最大的动作
            actions_value = self.eval_net(observation)
            # index = np.argmax(actions_value)
            index = torch.max(actions_value, 1)[1].data.numpy()
            index = index[0]
            action = actions[index]
        else:
            index = np.random.randint(0, self.n_actions)
            action = actions[index]
        return action


    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 直接赋值更新权重
            print('\ntarget_params_replaced\n')
        self.learn_step_counter += 1

        # 随机抽样
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(batch_memory[:, : self.n_features])
        b_a = torch.LongTensor(batch_memory[:, self.n_features:self.n_features + 1].astype(int))
        b_r = torch.FloatTensor(batch_memory[:, self.n_features+2])
        b_s_ = torch.FloatTensor(batch_memory[:, -self.n_features:])

        # eval中的参数
        q_target = torch.zeros((self.batch_size, 1))
        q_eval = self.eval_net(b_s)
        q = q_eval
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        for i in range(b_s.shape[0]):
            action = torch.argmax(q[i], 0).detach()
            q_target[i] = b_r[i] + self.gama * q_next[i, action]

        # print("q_eval:{}\nq_target:{}".format(q_eval, q_target))
        loss = self.loss_fun(q_eval, q_target)

        self.cost_his.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # print("epsilon:", self.epsilon)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        cost_ = self.cost_his
        for i in range(self.epoch):
            cost_.remove(max(cost_))
        plt.plot(np.arange(len(cost_)), cost_)
        plt.ylabel('Loss')
        plt.xlabel('training steps')
        plt.show()

        # 保存专家数据
        # np.savetxt('memory.csv', self.memory, fmt='%.2f', delimiter=',')









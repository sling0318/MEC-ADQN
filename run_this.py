from RL_brain import DeepQNetwork
import numpy as np

EPOCH = 100

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

n_actions = len(actions)
n_features = 32
lam_local, beta_local, cycle_perbyte, energy_per_l= 0.6, 0.4, 1, 6
lam_re, beta_re, energy_per_r ,discount = 0.8, 0.2, 0.3, 0.01
local_core_max, local_core_min = 200, 50
server_core_max, server_core_min = 400, 150
uplink_max, uplink_min = 350, 100
downlink_max, downlink_min = 600, 250

def reset():
    np.random.seed(np.random.randint(1, 1000))
    workload = np.random.randint(2000, 3000)                                               # 定义工作来量
    local_comp = np.random.randint(90, 110)                                                # 本地计算资源
    uplink = np.array([np.random.randint(150, 200), np.random.randint(150, 200),
                       np.random.randint(150, 200), np.random.randint(150, 200),
                       np.random.randint(150, 200), np.random.randint(150, 200),
                       np.random.randint(150, 200), np.random.randint(150, 200),
                       np.random.randint(150, 200), np.random.randint(150, 200)])          # 定义初始上行链路容量
    downlink = np.array([np.random.randint(300, 500), np.random.randint(300, 500),
                         np.random.randint(300, 500), np.random.randint(300, 500),
                         np.random.randint(300, 500), np.random.randint(300, 500),
                         np.random.randint(300, 500), np.random.randint(300, 500),
                         np.random.randint(300, 500), np.random.randint(300, 500)])        # 定义下行链路容量
    servers_cap = np.array([np.random.randint(200, 300), np.random.randint(200, 300),
                            np.random.randint(200, 300), np.random.randint(200, 300),
                            np.random.randint(200, 300), np.random.randint(200, 300),
                            np.random.randint(200, 300), np.random.randint(200, 300),
                            np.random.randint(200, 300), np.random.randint(200, 300)])     # 定义服务器的可用计算资源，服务器数量为10
    observation = np.array([workload, local_comp])
    return np.hstack((observation, servers_cap, uplink, downlink))

def mec_step(observation, action, time1):
    workload, local_comp, servers_cap, uplink, downlink = \
        observation[0],observation[1],observation[2:12], observation[12:22], observation[22:32]
    target_server, percen = int(action[0]), action[1]
    wait_local, wait_server = 2, 1  # 本地与服务器的排队的任务大小

    local_time = lam_local * workload * cycle_perbyte * (1 - percen)/(local_comp) + wait_local * cycle_perbyte

    local_energy = beta_local * workload * (1 - percen)

    local_only = lam_local * workload * cycle_perbyte/(local_comp) + wait_local * cycle_perbyte + beta_local * workload


    remote_time = lam_re * workload * cycle_perbyte * percen / (servers_cap[target_server]) + wait_server * cycle_perbyte + \
                    workload * percen / (uplink[target_server]) + discount * workload / (downlink[target_server])

    remote_energy = beta_re * workload * percen

    time_cost = local_time + remote_time

    energy_cost = local_energy + remote_energy

    remote_only = lam_re * workload * cycle_perbyte / (servers_cap[target_server]) + wait_server * cycle_perbyte + \
                    workload / (uplink[target_server]) + discount * workload / (downlink[target_server]) + \
                    beta_re * workload

    total_cost = 0.6 * time_cost + 0.4 * energy_cost
    reward = -total_cost

    np.random.seed(np.random.randint(1, 10000))

    # 建立下一个过程的模拟生成
    a = np.random.uniform()
    b = 0.9
    if(time1 >= 0) and (time1 <= 36):
        if(a > b):
            local_comp = min(local_comp + np.random.randint(0, 6), local_core_max)
            for i in range(4):
                servers_cap[i] = min(servers_cap[i] + np.random.randint(0, 15), server_core_max)
                downlink[i] = min(downlink[i] + np.random.randint(0, 8), downlink_max)
                uplink[i] = min(uplink[i] + np.random.randint(0, 5), uplink_max)

        else:
            local_comp = max(local_comp + np.random.randint(-5, 0), local_core_min)
            for i in range(4):
                servers_cap[i] = max(servers_cap[i] + np.random.randint(-14, 0), server_core_min)
                downlink[i] = max(downlink[i] - np.random.randint(0, 8), downlink_min)
                uplink[i] = max(uplink[i] - np.random.randint(0, 5), uplink_min)
        workload += np.random.randint(-100, 200)

    elif (time1 > 36) and (time1 <= 72):
        if (a < b):
            local_comp = min(local_comp + np.random.randint(0, 6), local_core_max)
            for i in range(4):
                servers_cap[i] = min(servers_cap[i] + np.random.randint(0, 15), server_core_max)
                downlink[i] = min(downlink[i] + np.random.randint(0, 8), downlink_max)
                uplink[i] = min(uplink[i] + np.random.randint(0, 5), uplink_max)

        else:
            local_comp = max(local_comp + np.random.randint(-5, 0), local_core_min)
            for i in range(4):
                servers_cap[i] = max(servers_cap[i] + np.random.randint(-14, 0), server_core_min)
                downlink[i] = max(downlink[i] - np.random.randint(0, 8), downlink_min)
                uplink[i] = max(uplink[i] - np.random.randint(0, 5), uplink_min)
        workload += np.random.randint(-200, 100)


    elif (time1 > 72) and (time1 <= 108):
        if (a > b):
            local_comp = min(local_comp + np.random.randint(0, 6), local_core_max)
            for i in range(4):
                servers_cap[i] = min(servers_cap[i] + np.random.randint(0, 15), server_core_max)
                downlink[i] = min(downlink[i] + np.random.randint(0, 8), downlink_max)
                uplink[i] = min(uplink[i] + np.random.randint(0, 5), uplink_max)

        else:
            local_comp = max(local_comp + np.random.randint(-5, 0), local_core_min)
            for i in range(4):
                servers_cap[i] = max(servers_cap[i] + np.random.randint(-14, 0), server_core_min)
                downlink[i] = max(downlink[i] - np.random.randint(0, 8), downlink_min)
                uplink[i] = max(uplink[i] - np.random.randint(0, 5), uplink_min)
        workload += np.random.randint(-100, 200)
    observation_ = np.array([workload, local_comp])
    observation_1 = np.hstack((observation_, servers_cap, uplink, downlink))
    return observation_1, reward, local_only, remote_only

def run_mec_offloading():
    step = 0
    local_only_cost, remote_only_cost, total_cost = [], [], []
    for epoch in range(EPOCH):

        observation = reset()

        for time_1 in range(108):
            action = RL.choose_action(observation)
            observation_, reward, local_only, remote_only = mec_step(observation, action, time_1)
            RL.store_transition(observation, action, reward, observation_)

            if(step > 200) and (step % 5 == 0):
            # if RL.memory_counter > RL.memory_size:
                RL.learn()
            if step > 2000 and step % 100 == 0:
                local_only_cost.append(local_only)
                remote_only_cost.append(remote_only)
                total_cost.append(-reward)

            observation = observation_
            step += 1
        print("epoch:{}|||reward:{}".format(epoch, reward))

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(local_only_cost)), local_only_cost, 'b')
    plt.plot(np.arange(len(remote_only_cost)), remote_only_cost, 'g')
    plt.plot(np.arange(len(total_cost)), total_cost, 'r')
    plt.legend(("Execute_Local", "Execute_Remote", "Advanced_DQN"))
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()
    # end of game
    print('game over')


if __name__ == '__main__':
    RL = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      epoch=EPOCH,
                      output_graph=True
    )
    run_mec_offloading()
    RL.plot_cost()
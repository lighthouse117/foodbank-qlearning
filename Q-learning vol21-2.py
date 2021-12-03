# coding:utf-8
# Reinforcement Learning Q-learning 強化学習　Q学習
# Optimization of Logistics 物流最適化
# 拠点数7
# 状態数＝802,816（5*4096）, 行動数=7
# 状態定義
# ＝　現在いる拠点
# + [要求が満たされたか，他の拠点が欲している食品を持っているか]（2bit×5拠点）（10bit＝1024通り）
# + 現在の1ステップ前にいた拠点
# 報酬は全配送終了後+1，かつ現在の最短距離と等しいまたはより短い場合は+10
# 拠点の要求が満たされたかどうかのフラグ + 他拠点が欲している余りものがあるかどうかのフラグ
# シンプルな食品設定
# 全配送完了後，暗黙的に0に戻らせる(距離を足す)のではなく，エージェントに実際に0拠点に行動させる
# 目標状態で報酬を与える際は，目標値は報酬のみ
# 結果を得るためのgreedyでは学習させない
# ε,αを時間経過とともに減衰
# 結果を出力

import random
import os
import csv
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


DIR_PATH = os.path.dirname(os.path.abspath(__file__))


# 割引率，未来の報酬の価値(Discount factor, Worth of future rewards)
GAMMA = 0.6
ALPHA = 0.05                         # 学習率(Learning rate)
# ε，ランダムに探索をする確率の初期値(Initial value of probabiolity of random exploration)
EPSILON = 1
MAX_EPISODES = 200000                # 最大エピソード数(Maximum number of episodes)
MAX_STEPS = 50                      # 最大ステップ数(Maximum number of steps)

MIN_ALPHA = 0
MIN_EPSILON = 0.8

GREEDY_CYCLE = 500

N = 7                               # 都市数(Number of cities)


foods = {           # List of foods   [name, weight]
    "tomato": 10,
    "banana": 15,
    "apple": 20,
}


MAX_CAPACITY = 10   # Maximum capacity of track


stocks = [
    {   # city0
        "tomato": 0,
    },
    {   # city1
        "tomato": 0,
    },
    {   # city2
        "tomato": 0,
    },
    {   # city3
        "tomato": 0,
    },
    {   # city4
        "tomato": 0,
    },
    {   # city5
        "tomato": 0,
    },
    {   # city6
        "tomato": 200,
    },
]

surpluses = [    # List of surplus foods in each city [name, quantity]
    {   # city0
    },
    {},
    {},
    {   # city3
    },
    {},
    {},
    {
        "tomato": 100,
    },
]


requirements = [    # List of foods required in each city [name, quantity]
    {"tomato": 5},  # city0
    {"tomato": 5},  # city1
    {"tomato": 5},  # city2
    {"tomato": 5},  # city3
    {"tomato": 5},  # city4
    {"tomato": 5},
    {},
]

foods_in_cargo = {}

surpluse_status = [0 for _ in range(N)]
requirement_status = [0 for _ in range(N)]

required_food = []

temp_stocks = []
temp_requirements = []
temp_surpluses = []


class QLearning:
    def __init__(self):
        # Q値，行動価値関数(Q-value, Action-value function)
        status_size = N * N * pow(4, N)
        # print("STATUS_SIZE", status_size)
        self.Q = np.zeros((status_size, N))
        # 都市のxy座標(XY coordinates of cities)
        self.cities = []
        # 訪れた都市のルート(Route of visited cities)
        self.route = []
        # 最短ルート(The shortest route)
        self.min_route = []
        # 最短ルートの距離(Distace of the shortest route)
        self.min_distance = 1000000
        self.min_distance_history = []

        self.min_state = 0

        self.greedy_min_distance = 1000000

        self.reward_history = []
        self.greedy_distance = []
        self.greedy_route = []
        self.remaining_city = []

        self.result = []

        np.set_printoptions(suppress=True)

    def load_city(self):
        # 都市の座標をファイルから読み込み(Load the coordinates of cities from file)
        with open(os.path.join(DIR_PATH, "city7.txt"), "r") as city_file:
            city_reader = csv.reader(city_file)
            self.cities = [[int(c) for c in row] for row in city_reader]

    def run(self):

        global EPSILON, ALPHA
        global temp_stocks, temp_requirements, temp_surpluses, requirement_status, surpluse_status

        distance_and_route = {}

        max_d = 0

        ALPHA = 0.05
        EPSILON = 1

        alpha_step = (ALPHA - MIN_ALPHA) / (MAX_EPISODES * 0.9)
        epsilon_step = (EPSILON - MIN_EPSILON) / (MAX_EPISODES * 0.9)

        mode = 0
        all_distance = 0

        # print("STOCKS : {}".format(stocks))

        # エピソードごとのループ(Start loop for an episode)
        for i in range(MAX_EPISODES + 1):

            step_count = 0
            reward_per_episode = 0
            finish_flag = False
            go_back_flag = False
            # print("------Episode: {}------".format(i))

            self.initialize_cargo()
            self.visited = [0 for _ in range(N)]

            temp_stocks = deepcopy(stocks)
            temp_requirements = deepcopy(requirements)
            temp_surpluses = deepcopy(surpluses)
            requirement_status = [0 for _ in range(N)]
            surpluse_status = [0 for _ in range(N)]

            # 状態に出発点を設定(Give start position of cities(state))
            current_city = 0

            self.load_and_unload_foods(current_city, mode)
            self.check_delivery_status()

            state_status = 0
            for digit, (flag_r, flag_s) in enumerate(zip(requirement_status, surpluse_status)):
                if flag_r == 1:
                    state_status += pow(2, 2 * digit)
                if flag_s == 1:
                    state_status += pow(2, 2 * digit + 1)

            state = pow(4, N) * (N * current_city +
                                 current_city) + state_status

            # 出発都市をルートに追加(Add start city to the route)
            self.route.append(current_city)

            if i % GREEDY_CYCLE == 0:
                mode = 1
                # print()
                print("----------Episode: {}-----------".format(i))
                # print("MODE1")
                # print(self.Q.astype(float))
                # print()
                # print("STATE ===> {}".format(state))
            else:
                mode = 0

            # ステップごとのループ(Start loop for each step)
            while True:

                # print("---Step : {}---".format(j))

                if mode == 1:
                    if go_back_flag is True:
                        action = 0
                    else:
                        action = self.choose_action_greedy(state, current_city)
                    # print("CURRENT CITY ===> {}".format(current_city))
                    # print("STATE ===> {}".format(state))
                    # print("REQUIREMENT STATUS ===> {}".format(requirement_status))
                    # print("SURPLUS STATUS ===> {}".format(surpluse_status))
                    # print("ACTION ===> {}".format(action))
                    # print()
                else:
                    # ε-greedy法を用いて行動(次の都市)を選択(Choose next action(destination) by using epsilon-greedy algorithm)
                    if go_back_flag is True:
                        action = 0
                    else:
                        action = self.choose_action_epsilon_greedy(
                            state, current_city)

                # print("STATE : {}".format(state))
                # print("ACTION ===> {}".format(action))
                # print("ROUTE : {}".format(self.route)

                self.load_and_unload_foods(action, mode)
                self.check_delivery_status()

                state_status = 0
                for digit, (flag_r, flag_s) in enumerate(zip(requirement_status, surpluse_status)):
                    if flag_r == 1:
                        state_status += pow(2, 2 * digit)
                    if flag_s == 1:
                        state_status += pow(2, 2 * digit + 1)

                next_state = pow(4, N) * (N * action +
                                          current_city) + state_status

                # 選択した都市をルートに追加(Add selected city to the route)
                self.route.append(action)

                # if state == 163839:
                #     print(self.route)
                #     print(next_state)

                reward = 0

                # print("REWARD(DISTANCE) : {}".format(reward))
                # reward_per_episode += reward

                if all(requirement_status) is True:
                    go_back_flag = True
                    if action == 0:
                        finish_flag = True
                        reward = 1

                        all_distance = self.calcDistance()

                        # d = [343, 406, 413]
                        # if all_distance in d:
                        #     print(all_distance, self.route)

                        if len(self.route) == 8:
                            if all_distance > max_d:
                                print(all_distance, self.route)
                                max_d = all_distance

                        if all_distance <= self.min_distance:
                            reward = 10
                            # print(reward)
                            # print("Found : ", all_distance)
                            self.min_distance = all_distance
                            self.min_route = deepcopy(self.route)
                            self.min_state = state

                if step_count >= MAX_STEPS:
                    all_distance = None
                    finish_flag = True
                    # reward = -1

                if mode != 1:
                    if all(requirement_status) is True and finish_flag is True:
                        self.learn_goal(state, action, reward)
                    else:
                        # Q値を更新(Update Q-value)
                        self.learn(state, action, reward, next_state)

                # print(self.Q.astype(float))

                # 全ての制約を満たしたとき(When all constraints are satisfied)

                if finish_flag is True:
                    if ALPHA > MIN_ALPHA:
                        ALPHA -= alpha_step
                    else:
                        ALPHA = MIN_ALPHA

                    if EPSILON > MIN_EPSILON:
                        EPSILON -= epsilon_step
                    else:
                        EPSILON = MIN_EPSILON

                    if mode == 1:
                        # print("ROUTE : {}".format(self.route))
                        # print("DISTANCE : {}".format(all_distance))
                        # print("EPSILON : {}".format(EPSILON))
                        # print("ALPHA : {}".format(ALPHA))
                        # print("MIN DISTANCE : {}".format(self.min_distance))
                        # print("MIN STATE : {}".format(self.min_state))
                        # print("163839 : {}".format(self.Q[163839][0]))
                        # print("REWARD : {}".format(reward))
                        # print("MIN ROUTE : {}".format(self.min_route))
                        # print()
                        self.result.append([i, all_distance, EPSILON, ALPHA])
                        self.greedy_distance.append(all_distance)

                    # print("ROUTE : {}".format(self.route))

                    # 次のエピソードに進む(Move on to next episode)
                    break

                # 状態を更新(Replace current state with next state)
                state = next_state
                current_city = action
                step_count += 1

            # ルートを初期化(Initialize route)
            self.route = []
            # self.remaining_ship = deepcopy(self.ship)               # remaining_shipを初期化(Initialize remaining_ship)
            self.min_distance_history.append(self.min_distance)
            self.reward_history.append(reward_per_episode)

            # print(self.Q.astype(float))
            # if i == 70000:
            #     break

        # print()
        # print("Route : {}".format(self.min_route))
        # print("Distance : {}".format(self.min_distance))
        # print(self.Q.astype(float))
        # print(self.Q[:1024].astype(float))
        # self.normalize_Q()
        # print(self.Q.astype(int))

        # for i in range(1024):
        #     if np.any(self.Q[i]):
        #         print(i, self.Q[i])

        # print(555, self.Q[555])
        # print(21291, self.Q[21291])
        # print(16367, self.Q[16367])
        # print(11262, self.Q[11262])
        # print()
        # print(815, self.Q[815])
        # print(5935, self.Q[5935])
        # print(11055, self.Q[11055])

        # print("MIN ROUTE : {}".format(self.min_route))
        # print("MIN DISTANCE : {}".format(self.min_distance))
        # print("STOCKS : {}".format(temp_stocks))
        # print(self.greedy_route)
        print(distance_and_route)

        self.save_result()

        # self.print_reward_graph()
        # self.print_distance_graph()

    def choose_action_epsilon_greedy(self, state, current_city):
        global requirement_status, surpluse_status
        action = current_city
        if np.random.rand() <= EPSILON:
            # while action not in self.remaining_city:
            while action == current_city:
                action = random.randint(0, N - 1)
            return action
        else:
            i = 0
            sortedQ = np.argsort(self.Q[state])[::-1]
            while action == current_city:
                action = sortedQ[i]
                i += 1
                if i >= N - 1:
                    break
            return action

    def choose_action_greedy(self, state, current_city):
        global requirement_status, surpluse_status
        action = current_city
        i = 0
        sortedQ = np.argsort(self.Q[state])[::-1]
        while action == current_city:
            action = sortedQ[i].astype(int)
            i += 1
            if i >= N - 1:
                break
        return action

    def learn(self, state, action, reward, next_state):
        # Q値の最大値(Maximum value of Q-value)
        sorted_next_Q = np.sort(self.Q[next_state])[::-1]
        max_next_Q = 0
        i = 0
        while max_next_Q == 0:
            max_next_Q = sorted_next_Q[i]
            i += 1
            if i == N:
                break

        predict = self.Q[state, action]
        target = reward + GAMMA * max_next_Q

        self.Q[state, action] += ALPHA * \
            (target - predict)     # Q値を更新(Update Q function)
        # print(self.Q[state, action])

    def learn_goal(self, state, action, reward):
        # if state == self.min_state and reward != 10:
        #     return

        predict = self.Q[state, action]
        target = reward

        self.Q[state, action] += ALPHA * \
            (target - predict)     # Q値を更新(Update Q function)

        # if reward == 10:
        #     print(state, action, predict)

        # print(self.Q[state, action])

    def calcDistance(self):

        # 距離を初期化(Initialize distance)
        distance = 0

        # 最初の都市のx座標(X coordinate of the first city)
        start_x = self.cities[self.route[0]][0]
        # 最初の都市のy座標(Y coordinate of the first city)
        start_y = self.cities[self.route[0]][1]
        # 最初の都市のベクトルを生成(Generate vector of the first city)
        start = np.array([start_x, start_y])

        # ルートのループ(Start loop for route)
        for i in range(len(self.route)):

            # ルートの末尾でないなら(If it is not the end of route)
            if i != len(self.route) - 1:
                # 次の都市のx座標(X coordinate of the destination)
                destination_x = self.cities[self.route[i + 1]][0]
                # 次の都市のy座標(Y coordinate of the destination)
                destination_y = self.cities[self.route[i + 1]][1]
                # 次の都市のベクトルを生成(Generate vector of the destination)
                destination = np.array([destination_x, destination_y])

                # 2点間のベクトルの距離を求めて加算(Calculate and add distance of vector between 2 cities)
                distance += int(np.linalg.norm(destination - start))

                # 今の目的地を次のスタート地点に更新(Replace start with destination)
                start = destination.copy()

            else:
                break

        return distance

    def initialize_cargo(self):
        global foods_in_cargo, required_food
        for name in foods:
            foods_in_cargo[name] = 0
        required_food = []
        # print("CURRENT CARGO = {}".format(foods_in_cargo))

    def calcVolume(self):
        global foods_in_cargo
        volume = 0
        for name, quantity in foods_in_cargo.items():
            volume += quantity * foods[name]
        return volume

    def load_and_unload_foods(self, base, mode):
        global foods_in_cargo, temp_stocks, temp_requirements, temp_surpluses

        if mode == 1:
            # print("STOCK OF CITY{} = {}".format(base), temp_stocks[base]))
            # print("REQUIREMENTS OF CITY{} = {}".format(base, temp_requirements[base]))
            pass

        # unload
        for name, require_quantity in list(temp_requirements[base].items()):

            if foods_in_cargo[name] == 0 or require_quantity == 0:
                continue

            elif require_quantity <= foods_in_cargo[name]:
                if mode == 1:
                    # print("----Unloading {} {}s.----".format(require_quantity, name))
                    pass
                temp_stocks[base][name] += require_quantity
                temp_requirements[base][name] = 0
                foods_in_cargo[name] -= require_quantity

            else:
                temp_stocks[base][name] += foods_in_cargo[name]
                temp_requirements[base][name] -= foods_in_cargo[name]
                foods_in_cargo[name] = 0

        # load
        for name, surpluse_quantity in list(temp_surpluses[base].items()):
            if surpluse_quantity == 0:
                continue

            if mode == 1:
                # print("----Loading {} {}s.----".format(surpluse_quantity, name))
                pass

            # current_volume = self.calcVolume()
            # if MAX_CAPACITY < current_volume + foods[name]:

            temp_stocks[base][name] -= surpluse_quantity
            foods_in_cargo[name] += surpluse_quantity
            temp_surpluses[base][name] = 0

            # volume = self.calcVolume()

        # if mode == 1:
            # print("STOCK OF CITY{} = {}".format(base, temp_stocks[base]))
            # print("CURRENT CARGO = {}".format(foods_in_cargo))
            # print("CURRENT VOLUME = {}".format(volume))

    def check_delivery_status(self):
        global temp_requirements, temp_surpluses, requirement_status, surpluse_status, required_food

        for base, r in enumerate(temp_requirements):
            requirement_status[base] = 1
            for name, quantity in r.items():
                if quantity >= 1:
                    requirement_status[base] = 0
                    if name not in required_food:
                        required_food.append(name)
                else:
                    if name in required_food:
                        required_food.remove(name)

        for base, s in enumerate(temp_surpluses):
            surpluse_status[base] = 1
            for name, quantity in s.items():
                if quantity >= 1 and name in required_food:
                    surpluse_status[base] = 0
                    break

    def normalize_Q(self):
        max_Q = self.Q.max()
        if max_Q == 0:
            return
        rate = 100.0 / max_Q
        self.Q = self.Q * rate

    def print_distance_graph(self):
        x = np.array(
            [i * GREEDY_CYCLE for i in range(len(self.greedy_distance))])
        y = np.array(self.greedy_distance)
        plt.plot(x, y)
        plt.ylim(0,)
        # plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Distance")
        # plt.text(800, self.min_distance_history[0], "ε={}".format(EPSILON))
        plt.show()
        # plt.savefig(os.path.join(DIR_PATH, "img/tsp{}".format(self.generation)))
        return

    def save_result(self):
        with open(os.path.join(DIR_PATH, "result/result.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Distance"])
            writer.writerows(self.result)


if __name__ == '__main__':
    success_count = 0
    failed_count = 0
    distance_history = []

    for i in range(1):
        # print("---- Trial:{} ----".format(i))
        QL = QLearning()

        QL.load_city()

        QL.initialize_cargo()

        QL.run()

        distance = QL.greedy_distance[-1]

        print(distance)
        distance_history.append(distance)

        if QL.greedy_distance[-1] == 263:
            success_count += 1
        else:
            failed_count += 1

    print("SUCCESS : ", success_count)
    print("FAILED : ", failed_count)

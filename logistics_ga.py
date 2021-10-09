# coding:utf-8
# 物流最適化(TSPに配達順序制約を追加)　遺伝的アルゴリズム

import random, sys, os, csv
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

SUCCESS = 1                 # 成功
FAIL = 0                    # 失敗

N = 5                      # 都市数

GENERATION_MAX = 200        # 最大世代
POPULATION = 100            # 個体数(集団サイズ)
CROSS_NUM = 50              # 交叉時の親のペア数

ELITE_RATE = 10             # エリート選択を行う上位個体の割合[%]
CROSS_PROBABILITY = 80     # 交叉確率
MUTATION_RATE = 5           # 突然変異率

REST_MAX = 10               # 最大初期化回数

class Logistics:
    def __init__(self):
        self.genes = [[] * N for _ in range(POPULATION)]  # 個体(遺伝子)
        self.fitness = [0 for _ in range(POPULATION)]       # 適応度
        self.generation = 0                                 # 世代
        self.rank_index = [0 for _ in range(POPULATION)]    # 個体を適応度順に並び替えたインデックス
        self.city = []                                       # 都市のxy座標
        self.ship = []

    def generate_city(self):
        '''
        ### 乱数で都市を初期生成してself.cityにxy座標を保存
        '''
        x_range = 100   # x座標の最大値
        y_range = 100   # y座標の最大値
        x_coordinate = [random.randint(0, x_range) for _ in range(N)]   # x座標をランダムに生成
        y_coordinate = [random.randint(0, y_range) for _ in range(N)]   # y座標をランダムに生成
        self.city = [[x_coordinate[i], y_coordinate[i]] for i in range(N)]  # x,y座標を格納する

    
    def save_city(self):
        with open(os.path.join(DIR_PATH, "city.txt"), "w") as city_file:
            city_writer = csv.writer(city_file, lineterminator="\n")
            city_writer.writerows(self.city)
    
    def load_city(self):
        with open(os.path.join(DIR_PATH, "city5.txt"), "r") as city_file:
            city_reader= csv.reader(city_file)
            self.city = [[int(c) for c in row] for row in city_reader]
    
    def set_ship(self): 
        # 配達元，配達先を設定
        # self.ship.append([1, 3])
        # self.ship.append([4, 2])
        return

    def create_initial_population(self):
        '''
        ### 乱数で初期集団(遺伝子)を生成してself.genesに保存
        '''
        for i in range(POPULATION):
            # 0～Nまでのリストをランダムに並び替えた個体を生成
            # 出発点は0とする
            self.genes[i].append(0)
            # 出発地を除いた残りの拠点をランダムに並び替え個体とする
            self.genes[i].extend(random.sample(range(1, N), N-1))   
        # print(self.genes)


    def calc_fitness(self):
        '''
        ### 各個体の遺伝子の適応度(移動コスト)を評価して適応度をself.fitnessに保存
        ### 適応度＝全都市間の座標のユークリッド距離
        ### 適応度が低いほど優秀な遺伝子
        '''
        self.fitness = [0 for _ in range(len(self.genes))] # 適応度を初期化
        # 各個体について全都市間の移動コストを加算する
        for i in range(len(self.genes)):
            # 0個目の都市をスタート地点としてベクトルを生成
            start_x = self.city[self.genes[i][0]][0]
            start_y = self.city[self.genes[i][0]][1]
            start = np.array([start_x, start_y])
            for j in range(N):
                if j == N-1:
                    # 最後の都市(リストの最後)は最初の都市との距離を計算
                    destination_x = self.city[self.genes[i][0]][0]
                    destination_y = self.city[self.genes[i][0]][1]
                    destination = np.array([destination_x, destination_y])
                    # 2点間のベクトルの距離を求めて適応度に加算
                    self.fitness[i] += int(np.linalg.norm(destination-start))
                else:
                    # 都市j+1を目的地としてベクトルを生成
                    destination_x = self.city[self.genes[i][j+1]][0]
                    destination_y = self.city[self.genes[i][j+1]][1]
                    destination = np.array([destination_x, destination_y])
                    # 2点間のベクトルの距離を求めえ適応度に加算
                    self.fitness[i] += int(np.linalg.norm(destination-start))
                    # 今の目的地を次のスタート地点に更新
                    start = destination.copy()
            
            # 配達順序を調べる
            for ship in self.ship:
                    if ship[0] == 0 or ship[1] == 0:    # 配達元or配達先が0の場合は配達順序を必ず満たしているためスキップ
                        continue
                    if self.genes[i].index(ship[0]) > self.genes[i].index(ship[1]):   # 遺伝子が配達順序を満たしていない場合(配達元より配達先が前にある場合)
                        # 適応度を加算
                        self.fitness[i] += 1000

            

    def select(self):
        '''
        ### 個体の淘汰(選択)
        ### 適応度の高い上位[ELITE_RATE]%はエリート選択で保持し，残りの個体はルーレット選択により抽出
        '''
        # 遺伝子をコピー
        old_genes = deepcopy(self.genes)            

        # エリート保存を行う
        elite_num = int(POPULATION * ELITE_RATE * 0.01)  # エリート選択を行う上位個体の数
        self.rank_index = sorted(range(len(self.genes)), key=lambda k: self.fitness[k])  # 遺伝子を適応度順に並び替えたインデックス
        for i in range(elite_num):
            self.genes[i] = old_genes[self.rank_index[i]]                           # 適応度が上位の個体を次世代に保持

        
        # ルーレット選択を行う
        roulette_fitness = []   # 残りのルーレット選択をする個体の適応度を格納
        for i in range(elite_num, len(self.genes)):
            roulette_fitness.append(self.fitness[self.rank_index[i]])
        fitness_total = 0       # 適応度の和(正規化用)
        for i in roulette_fitness:
            fitness_total += i
        # ルーレットの的を作る(正規化された選択確率)
        choice_probability = [i/fitness_total for i in roulette_fitness]    # 各適応度を適応度の和で割ったものを選択確率とする
        # ルーレットを回す
        # 残りの個体を選択確率によって抽出し，そのインデックスを得る
        roulette_count = POPULATION - elite_num
        roulette_result = np.random.choice(range(elite_num, len(self.genes)), roulette_count, p=choice_probability)
        # 選ばれた個体を次世代へ移す 
        for i, j in zip(range(elite_num, POPULATION), roulette_result):
            self.genes[i] = old_genes[j]

        # 残りの個体は消去
        del self.genes[POPULATION:]



    def crossover(self):
        '''
        ### 個体を交叉させ，親から新しい個体(子供)を作る
        ### 重複を避けるため順序交叉を行う
        '''
        for i in range(CROSS_NUM):
            # 親のペアをランダムに選択する
            parents_index = np.random.choice(len(self.genes), 2, replace=False)
            # 交叉確率により交叉させるかを決定
            cross_prob = random.randint(0, 100)
            if cross_prob < CROSS_PROBABILITY:
                # 順序交叉を行う
                # 切断個所をランダムに決定
                cut_index = random.randint(1, N-1)
                # 切断個所より左側の遺伝子はそのまま受け継ぐ
                child1 = self.genes[parents_index[0]][:cut_index]
                child2 = self.genes[parents_index[1]][:cut_index]
                # 切断個所より右側のは相手の遺伝子の順序を受け継ぐ
                for i in self.genes[parents_index[1]]:
                    if i not in child1:
                        child1.append(i)
                for i in self.genes[parents_index[0]]:
                    if i not in child2:
                        child2.append(i)
            else:
                # 交叉させず親の個体を複製する
                child1 = self.genes[parents_index[0]][:]
                child2 = self.genes[parents_index[1]][:]
            self.genes.append(child1)
            self.mutate()
            self.genes.append(child2)
            self.mutate()


    def mutate(self):
        '''
        ### 突然変異率にしたがって突然変異を起こす
        ### 逆位：ランダムな2点間の順序を逆転する
        ### 交叉で子供が生成されるたびに呼び出される
        '''
        # 突然変異率
        mutation_prob = random.randint(0, 100)
        if mutation_prob < MUTATION_RATE:
            # 突然変異を起こす
            # print("Mutate")
            mutate_point = random.sample(range(1, N), 2)   # ランダムに2点を選ぶ
            mutate_point.sort()                         # 2点を小さい順に入れ替え
            # 2点間の遺伝子を抽出
            new_chromosome = self.genes[-1][mutate_point[0]:mutate_point[1]]
            # 切り離した遺伝子の順序を逆転させる
            new_chromosome.reverse()
            # 逆転させた遺伝子を元の個体に戻す
            self.genes[-1][mutate_point[0]:mutate_point[1]] = new_chromosome


    def print_route(self):
        '''
        ### ルートを描画
        '''
        plt.figure()
        x_coordinate = [self.city[i][0] for i in self.genes[0]]
        y_coordinate = [self.city[i][1] for i in self.genes[0]]
        x_coordinate.append(self.city[self.genes[0][0]][0])
        y_coordinate.append(self.city[self.genes[0][0]][1])
        plt.scatter(x_coordinate, y_coordinate)
        plt.plot(x_coordinate, y_coordinate, label=self.fitness[0])
        plt.title("Generation: {}".format(self.generation))
        plt.legend()
        plt.savefig(os.path.join(DIR_PATH, "img/tsp{}".format(self.generation)))
        plt.close()



    def run(self):
        # self.generate_city()
        # self.save_city()
        self.load_city()
        print(self.city)
        self.set_ship()
        self.create_initial_population()
        self.calc_fitness()             # 適応度を計算
        while True:
            if self.generation % 10 == 0:
                print("------Generation: {}------".format(self.generation))
                print("Fitness: {}".format(self.fitness[0]))
                self.print_route()          # 経路を描画
            if self.generation >= GENERATION_MAX:   # 指定の世代を超えた場合は終了
                self.print_route()
                print(self.genes[0])
                return     
            self.crossover()                # 交叉
            self.calc_fitness()             # 適応度を計算
            self.select()                   # 選択淘汰
            self.generation += 1            # 世代をインクリメント


if __name__ == '__main__':
    l = Logistics()
    l.run()
    sys.exit()

        # if q.run() == SUCCESS:
        #     q.print_queens(q.fitness.index(0))
        #     print("Fond solution.")
        #     print("generation : " + str(q.gereration))
        #     print("reset : " + str(reset_count))
        #     sys.exit()
        # if reset_count >= REST_MAX:
        #     print("Can't find solution.")
        #     sys.exit()
        # reset_count += 1

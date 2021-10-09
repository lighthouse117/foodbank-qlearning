# coding:utf-8
# Nクイーン問題　遺伝的アルゴリズム

import random
import sys
from copy import deepcopy

N = 10               # クイーンの数

SUCCESS = 1         # 成功
FAIL = 0            # 失敗

GENERATION_MAX = 10000  # 最大世代
REST_MAX = 10           # 最大初期化回数

POPULATION = 4      # 個体数

CROSS_INDEX1 = int((N-1) / 2)           # 交叉位置1
CROSS_INDEX2 = int((N-1) - ((N-1) / 4)) # 交叉位置2


class Queens:
    def __init__(self):
        self.genes = [[-1] * N for _ in range(POPULATION)]  # 個体(遺伝子)
        self.fitness = [0 for _ in range(POPULATION)]       # 適応度
        self.gereration = 0                                 # 世代
        self.rank_index = [0 for _ in range(POPULATION)]    # 個体を適応度順に並び替えたインデックス

    def create_initial_population(self):
        # 乱数で初期集団(遺伝子)を生成
        for cnt in range(POPULATION):
            # 個体を生成
            for a in range(N):
                # 列が被らないようにクイーンを置く
                while True:
                    initial = random.randint(0, N-1)
                    if not initial in self.genes[cnt]:
                        break
                self.genes[cnt][a] = initial

    def calc_fitness(self):
        # 各個体の遺伝子の適応度(制約違反数)を評価
        # 適応度＝制約違反している変数(クイーン)の数
        # 適応度=0のときが解である
        self.fitness = [0 for _ in range(POPULATION)] # 適応度を初期化
        for cnt in range(POPULATION):
            # 各個体について各a行のクイーンが他のクイーンと制約違反していないか調べる
            for a in range(N):
                b = self.genes[cnt][a]      # a行のクイーンの位置
                for i in range(N):          # 各行について
                    if i == a:              # i=aのときはとばす
                        continue
                    j = self.genes[cnt][i]              # i行のクイーンの位置
                    if b == j or abs(a-i) == abs(b-j):  # 縦の利き筋か斜めの利き筋になっている
                        self.fitness[cnt] += 1          # 制約違反数を+1
                        break                           # a行が制約違反していることが分かったのでもう調べる必要はない

    def select(self):
        # 個体の選択淘汰
        # 親の遺伝子を適応度順に並び替えたインデックス
        self.rank_index = sorted(
            range(POPULATION), key=lambda k: self.fitness[k])
        # 適応度の最も低い個体を適応度の最も高いエリート個体で置き換え，エリート個体を増殖させる
        # 適応度の低い個体は淘汰される
        self.genes[self.rank_index[POPULATION-1]
                   ] = deepcopy(self.genes[self.rank_index[0]])

    def crossover(self):
        # 個体を交叉させる
        child = [[-1] * N for _ in range(POPULATION)]  # 子供の個体を生成
        for i in range(POPULATION):
            child[i] = self.genes[self.rank_index[i]]  # 親の遺伝子をコピー
        # 交叉
        child[0][CROSS_INDEX1:], child[1][CROSS_INDEX1:] = child[1][CROSS_INDEX1:], child[0][CROSS_INDEX1:]
        child[2][CROSS_INDEX2:], child[3][CROSS_INDEX2:] = child[3][CROSS_INDEX2:], child[2][CROSS_INDEX2:]

    def mutation(self):
        # 突然変異を起こす
        gene_index = random.randint(0, POPULATION-1)        # どの個体を突然変異させるか
        # どの位置を突然変異させるか(クイーンを置く行)
        pos_index = random.randint(0, N-1)
        value_random = random.randint(0, N-1)               # ランダムな値(クイーンの位置)
        self.genes[gene_index][pos_index] = value_random    # 突然変異の実行

    def print_queens(self, a):
        for i in range(N):
            for j in range(N):
                if self.genes[a][i] == j:
                    print("Q ", end="")
                else:
                    print(". ", end="")
            print()
        print("-------------")

    def run(self):
        self.create_initial_population()
        while True:
            if self.gereration >= GENERATION_MAX:   # 指定の世代を超えた場合は失敗
                return FAIL
            self.calc_fitness()             # 適応度を計算
            # print(self.gereration)
            print(self.fitness)
            if 0 in self.fitness:           # 適応度が0の個体があった場合＝解を見つけた
                return SUCCESS
            self.select()                   # 選択淘汰
            self.crossover()                # 交叉
            self.gereration += 1            # 世代をインクリメント
            if self.gereration % 4 == 0:    # 4世代に1回の割合で突然変異
                self.mutation()


if __name__ == '__main__':
    reset_count = 0
    while True:
        q = Queens()
        if q.run() == SUCCESS:
            q.print_queens(q.fitness.index(0))
            print("Fond solution.")
            print("generation : " + str(q.gereration))
            print("reset : " + str(reset_count))
            sys.exit()
        if reset_count >= REST_MAX:
            print("Can't find solution.")
            sys.exit()
        reset_count += 1

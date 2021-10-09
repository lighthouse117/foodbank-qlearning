# coding:utf-8
# Nクイーン問題　制約違反最小化 山登り法

import random, sys

N = 10              # クイーンの数

SUCCESS = 1         # 成功
FAIL = 0            # 失敗

ITERATE_MAX = 100   # 最大反復回数
REST_MAX = 10       # 最大初期化回数


class Queens:
    def __init__(self):
        self.pos = [-1 for _ in range(N)]               # 各行のクイーンの位置
        self.is_conflict = [False for _ in range(N)]    # 各行が制約違反しているかどうか
        self.conflict_count = [0 for _ in range(N)]     # 制約違反数
        self.count = 0                                  # 反復回数

        # 乱数で初期値を生成
        for a in range(N):
            # 列が被らないようにクイーンを置く
            while True:
                initial = random.randint(0, N-1)
                if not initial in self.pos:
                    break
            self.pos[a] = initial

    def print_queens(self):
        for i in range(N):
            for j in range(N):
                if self.pos[i] == j:
                    print("Q ", end="")
                else:
                    print(". ", end="")
            print()
        print("-------------")

    def check_conflict(self):
        # 各a行のクイーンが他のクイーンと制約違反していないか調べる
        for a in range(N):
            self.is_conflict[a] = False  # どのクイーンとも制約違反していないとする
            b = self.pos[a]     # a行のクイーンの位置
            for i in range(N):  # 各行について
                if i == a:      # i=aのときはとばす
                    continue
                j = self.pos[i]  # i行のクイーンの位置
                if b == j or abs(a-i) == abs(b-j):  # 縦の利き筋か斜めの利き筋になっている
                    self.is_conflict[a] = True     # a行は制約違反している
                    break

    def count_conflict(self, a):
        # a行のクイーンを各列に動かした場合の制約違反数を調べる
        self.conflict_count = [0 for _ in range(N)]
        for b in range(N):      # b列に動かす
            for i in range(N):  # 各i行との制約違反
                j = self.pos[i]
                if b == j or abs(a-i) == abs(b-j):
                    self.conflict_count[b] += 1

    def run(self):
        while True:
            for i in range(len(self.is_conflict)):
                self.check_conflict()           # 制約違反を調べる
                if all([x == False for x in self.is_conflict]):  # 制約違反が0だった場合＝解を見つけた
                    return SUCCESS
                if self.count >= ITERATE_MAX:  # 指定の反復回数を超えた場合は失敗
                    return FAIL
                if self.is_conflict[i] == True:  # 制約違反をしているi行を選択
                    # print("iterate" + str(self.count))
                    self.count += 1 # 反復回数をインクリメント
                    # self.print_queens()
                    # print(self.is_conflict)
                    # print(i)
                    self.count_conflict(i)      # i行の各列の制約違反数を調べる
                    # print(self.conflict_count)
                    if min(self.conflict_count) >= self.conflict_count[self.pos[i]]:
                        # もしどこに動かしても現在の制約違反数から減らない場合は，別の行を選択
                        continue
                    self.pos[i] = self.conflict_count.index(
                        min(self.conflict_count))  # 制約違反数が最小になる値を選択


if __name__ == '__main__':
    reset_count = 0
    while True:
        q = Queens()
        if q.run() == SUCCESS:
            q.print_queens()
            print("Fond solution.")
            print("iteration : " + str(q.count))
            print("reset : " + str(reset_count))
            sys.exit()
        if reset_count >= REST_MAX:
            print("Can't find solution.")
        reset_count += 1

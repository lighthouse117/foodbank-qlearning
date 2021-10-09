# coding: utf-8
# 物流最適化テスト　バックトラック法

SUCCESS = 2     # 解である(成功)
OK = 1          # 現時点で制約違反はしていないが，まだ解ではない
CONFLICT = 0    # 制約違反している

N = 5           # 拠点数

BASE_A = 0      # 拠点A
BASE_B = 1      # 拠点B
BASE_C = 2      # 拠点C
BASE_D = 3      # 拠点D
BASE_E = 4      # 拠点E

food_weight = [5, 2, 3, 4, 3]   # 各食品の重量[kg]

base_distance = [               # 各拠点の距離[km]
    [0, 2, 2, 2, 2],
    [2, 0, 2, 2, 2],
    [2, 2, 0, 2, 2],
    [2, 2, 2, 0, 2],
    [2, 2, 2, 2, 0],
]

max_distance = 10               # 最大距離[km]
max_weight = 10                 # 最大積載量[kg]

class Logistics:
    def __init__(self):
        self.route = [0, 0]
        self.ship = []

    def set_ship(self):
    # 配達元，配達先，食品を設定
        self.ship.append([BASE_A, BASE_B, 0])  # A-B 食品0
        self.ship.append([BASE_A, BASE_C, 1])  # B-A 食品1
        self.ship.append([BASE_C, BASE_A, 2])  # A-C 食品2
        self.ship.append([BASE_A, BASE_D, 3])  # A-C 食品2
        self.ship.append([BASE_E, BASE_A, 3])  # A-C 食品2
        self.ship.append([BASE_D, BASE_C, 3])  # A-C 食品2
        self.ship.append([BASE_E, BASE_B, 3])  # A-C 食品2
        print(self.ship)


    def create_route(self):
    # 制約に違反しないよう拠点を追加していく
        while True:
            for i in range(N):
                self.route.insert(-1, i)                # 最後に拠点iを追加してみる
                result = self.check_constraint()        # 制約違反していないかチェック
                if result == SUCCESS:
                    return SUCCESS                      # 解を見つけたら終了
                elif result == OK:
                    if self.create_route() == SUCCESS:  # 制約違反していなければ次の拠点を追加
                        return SUCCESS             
                elif result == CONFLICT:
                    del self.route[-2]                  # 制約違反していた場合は追加した拠点を削除し，追加しなおす
            print("can't find solution")
            return


    def check_constraint(self):
    # 追加した拠点が制約違反していないか調べる
        print(self.route)

        # 拠点の重複を調べる
        for i in self.route[:-2]:
            if i == self.route[-2]:
                print("base is duplicate")
                return CONFLICT

        # 最大距離を調べる
        dist = 0
        for i in range(len(self.route)-1):
            # 各拠点間の距離を加算
            dist += base_distance[self.route[i]][self.route[i+1]]
        if dist > max_distance:
            print("exceed the distance limit")
            return CONFLICT

        # 配達順序を調べる
        for ship in self.ship:
            if self.route[-2] == ship[1]:           # 追加した拠点が配達先だったら
                if ship[0] not in self.route[:-2]:  # それより前に配達元がない場合制約違反
                    print("not satisfy ship order")
                    return CONFLICT

        # 全ての拠点を追加できたら成功，そうでないなら拠点の追加にもどる
        if set(range(N)) == set(self.route):
            print("found solution.")
            return SUCCESS
        else:
            print("no conflict.")
            return OK

    def run(self):
        self.set_ship()
        self.create_route()
        

if __name__ == '__main__':
    l = Logistics()
    l.run()
import numpy as np
import math
import matplotlib.pyplot as plt
import threading
from datetime import datetime


class Map:
    def __init__(self, ant_num):
        # 初始化城市距离
        self.distance_x = [
            178, 272, 176, 171, 650, 499, 267, 703, 408, 437, 491, 74, 532,
            416, 626, 42, 271, 359, 163, 508, 229, 576, 147, 560, 35, 714,
            757, 517, 64, 314, 675, 690, 391, 628, 87, 240, 705, 699, 258,
            428, 614, 36, 360, 482, 666, 597, 209, 201, 492, 294]
        self.distance_y = [
            170, 395, 198, 151, 242, 556, 57, 401, 305, 421, 267, 105, 525,
            381, 244, 330, 395, 169, 141, 380, 153, 442, 528, 329, 232, 48,
            498, 265, 343, 120, 165, 50, 433, 63, 491, 275, 348, 222, 288,
            490, 213, 524, 244, 114, 104, 552, 70, 425, 227, 331]
        self.city = len(self.distance_x)
        self.graph_dis = np.zeros((self.city, self.city))
        for i in range(self.city):
            for j in range(i, self.city):
                self.graph_dis[i][j] = pow(pow(self.distance_x[i]-self.distance_x[j], 2) +
                                           pow(self.distance_y[i]- self.distance_y[j], 2), 1/2)
                self.graph_dis[j][i] = self.graph_dis[i][j]
        self.num = ant_num
        self.open_table_city = [[True for _ in range(self.city)] for _ in range(self.city)]
        self.list_P = []  # 记录每只蚂蚁走过的节点和总共的距离
        for i in range(self.num):
            self.list_P.append([i])
            self.list_P[i].append(0)  # 代表走过的距离
        # 初始化信息素浓度
        self.message = np.ones((self.city, self.city))
        self.message[np.diag_indices_from(self.message)] = 0
        self.member = np.zeros((self.city))
        self.density = 0.5
        self.alpha = 1
        self.beta = 2
        self.steps = 1
        self.iteration = 500
        self.max = 0.5   # 前后变化的差值
        # 求取距离函数
        self.dis = np.zeros((self.iteration, self.num))
        self.best_dis = np.zeros((self.iteration))
        # 采取动作
        self.many_thread()

    def choose_next(self, ant, start_city):
        total = 0
        for i in range(self.city):
            if self.open_table_city[ant][i]:
                self.member[i] = pow(self.message[start_city][i], self.alpha) * pow(1 / self.graph_dis[start_city][i], self.beta)
                total += self.member[i]
        temp_pro = total * np.random.uniform(0, 1)
        for j in range(self.city):
            if self.open_table_city[ant][j]:
                temp_pro -= self.member[j]
                if temp_pro < 0:
                    next_city = j
                    break
        # 更新该蚂蚁走过的路程和距离
        self.list_P[ant].append((start_city, next_city))
        self.list_P[ant][1] += math.sqrt(pow(abs(self.distance_x[start_city]-self.distance_x[next_city]), 2) +
                                         pow(abs(self.distance_y[start_city]-self.distance_y[next_city]), 2))
        self.open_table_city[ant][next_city] = False
        return next_city

    def update_mess(self):
        # 更新所有可以走的路的信息素
        for start in range(self.city):
            for i in range(start + 1, self.city, 1):
                delta_mess = 0
                for ant in range(self.num):
                    target = i
                    route = {(start, target)}
                    route1 = {(target, start)}
                    if route.issubset(self.list_P[ant]) or route1.issubset(self.list_P[ant]):
                        delta_mess += 10 / self.list_P[ant][1]
                self.message[start][target] = (1 - self.density) * self.message[start][target] + delta_mess
                self.message[target][start] = self.message[start][target]

    def mainloop(self, ant_id, step):
        start_city = np.random.randint(0, self.city, 1)[0]
        origin = start_city
        self.open_table_city[ant_id] = [True for _ in range(self.city)]
        self.open_table_city[ant_id][start_city] = False
        self.list_P[ant_id] = []  # 记录每只蚂蚁走过的节点和总共的距离
        self.list_P[ant_id].append(ant_id)
        self.list_P[ant_id].append(0)  # 代表走过的距离
        # 这一部分是用来蚂蚁进行移动的，遍历所有城市
        for i in range(self.city - 1):
            n_city = self.choose_next(ant_id, start_city)
            start_city = n_city
        self.list_P[ant_id].append((start_city, origin))
        self.list_P[ant_id][1] += math.sqrt(pow(abs(self.distance_x[start_city] - self.distance_x[origin]), 2) +
                                         pow(abs(self.distance_y[start_city] - self.distance_y[origin]), 2))
        self.dis[step][ant_id] = self.list_P[ant_id][1]

    def many_thread(self):
        for i in range(self.iteration):
            threads = []
            for ant in range(self.num):
                threads.append(threading.Thread(target=self.mainloop(ant, i)))  # 同一进程下的线程之间共享进程内的数据 即用Thread合理
            for t in threads:
                t.start()
                t.join()
            if self.steps != 1:
                self.update_mess()
            self.steps += 1
            print(i)
            self.best_dis[i] = np.min(self.dis[i, :])
            # if i >= 2:
            #     if abs(2 * self.best_dis[i]-self.best_dis[i -1]- self.best_dis[i - 2]) < self.max:
            #         best_ant = np.where(self.dis == self.best_dis[i])[1]
            #         print(self.list_P[best_ant[0]])
            #         break
            # print('每步下的最短距离{}'.format(self.best_dis[i]))
        best_ant = np.argmin(self.dis[self.iteration - 1, :])
        path = []
        for i in range(self.city):
            path.append(self.list_P[best_ant][i + 2][0])
        print('闭环路径为：{}'.format(path))
        plt.plot(range(self.iteration), self.best_dis)
        plt.show()


a = Map(50)
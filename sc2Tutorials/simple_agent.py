# -*- coding: utf-8 -*-
# @Time    : 1/23/18 10:35 AM
# @Author  : Wang Zhaorong
# @Site    : 
# @File    : simple_agent.py
# @Software: PyCharm

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

# Functions
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index #SCREEN_FEATURES图层包含的player_relative图层的index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_BARRACKS = 21 #兵营
_TERRAN_COMMANDCENTER = 18 #人族指挥中心
_TERRAN_SUPPLYDEPOT = 19 #人族供给站
_TERRAN_SCV = 45 #农民

# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_NOT_QUEUED = [0]
_QUEUED = [1]


class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None #存储自己的中心坐标
    supply_depot_built = False #是否建供给
    scv_selected = False #是否选中农民
    barracks_built = False #是否建立兵营
    barracks_selected = False #是否选中兵营
    barracks_rallied = False #兵营集合
    army_selected = False #军队选中
    army_rallied = False #军队集合

    #坐标相对中心的转换
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        time.sleep(0.5)

        #中心点
        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        #建立供给
        if not self.supply_depot_built:
            #选中农民SCV，我们得到屏幕上所有SCV单元的x和y坐标，然后选择第一个x和y坐标[0]和unit_y[0]
            if not self.scv_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                target = [unit_x[0], unit_y[0]]

                self.scv_selected = True

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target]) #选中target坐标点

            #建立供应仓库，放在指挥中心附近，如果中心在左上角，则仓库放在中心下方20个坐标，中心在右下角则放在上方20个坐标处
            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

                self.supply_depot_built = True

                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])

        #构建兵营，放在指挥中心附近，如果中心在左上角，则仓库放在中心右侧20个坐标，中心在右下角则放在左侧20个坐标处
        elif not self.barracks_built:
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                self.barracks_built = True

                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        #陆战队：在训练陆战队之前，应该把兵营的集结点设置在坡道顶部，直到有足够的部队再离开，这样可以保护自己。
        elif not self.barracks_rallied:
            #选中兵营：取中心
            if not self.barracks_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

                #Return true if this collection contains any member that meets the given criterion.等到军营有空的时候
                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]

                    self.barracks_selected = True

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True

                if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])

                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])
        #如果供给有剩余并且农民可用，则构建陆战队
        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_MARINE in \
                obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        #送出军队
        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False

                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                self.army_selected = False

                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])

                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])


        return actions.FunctionCall(_NOOP, [])
from gbmj_share_func import PublicShareFun
from mahjong_ai import NetAi
import os
from os.path import join, abspath
from mahjong_player import Player
from mahjong_dealer import TileDealer
import random
from chupai_net import ChupaiPolicyValueNet as Net
from DEF import ACTION_TYPE, GAME_STATE, shape2_3, shape5, shape8
from utils import loose2values
import copy
import numpy as np
from handGroup import HandFanCal
import pdb
from datetime import datetime
import glob
from collections import OrderedDict
import re
import time
from read_n_write import write_to_pickle
import traceback
from random import shuffle

import matplotlib as mpl
mpl.use('Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from matplotlib import pyplot as plt
import pickle
import json
import os
the_changed_score = 1
base_chu = join(abspath('.'), 'model_files', 'SL_V5', 'chupai.model')
base_peng = join(abspath('.'), 'model_files', 'SL_V5', 'peng.model')
base_chow = join(abspath('.'), 'model_files', 'SL_V5', 'chow.model')
ai_dir = [base_chu, base_peng, base_chow]
chu_net_file = ai_dir[0]
peng_net_file = ai_dir[1]
chow_net_file = ai_dir[2]
# p_hulist = [0.0, 8.976660682226211e-06, 4.44558433786441e-05, 0.00015345815166281952, 0.0003941181499529794, 0.0009275882704967086, 0.0016982987090707019, 0.00269770026502522, 0.0037607933658202956, 0.004784560143626571, 0.005644182268957853, 0.006142173206805164, 0.006169530648884329, 0.005893818927930238, 0.0054137813114473795, 0.004735402239890571, 0.0038437206121227667, 0.0031418312387791743, 0.0024442164657604514, 0.0018205522783619732, 0.001345216722236471, 0.0008963836881251603, 0.0004612293750534325, 0.00015987005215012395, 3.505172266393092e-05, 5.984440454817474e-06, 4.2746003248696247e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
p_hulist = [0.0, 1.7953321364452423e-05, 8.89116867572882e-05, 0.00030691630332563903, 0.0007882362999059588, 0.0018551765409934172, 0.0033965974181414038, 0.00539540053005044, 0.007521586731640591, 0.009569120287253142, 0.011288364537915705, 0.012284346413610328, 0.012339061297768659, 0.011787637855860476, 0.010827562622894759, 0.009470804479781141, 0.007687441224245533, 0.0062836624775583485, 0.004888432931520903, 0.0036411045567239465, 0.002690433444472942, 0.0017927673762503206, 0.000922458750106865, 0.0003197401043002479, 7.010344532786184e-05, 1.1968880909634949e-05, 8.549200649739249e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# p_hulist = [0, 3.5906642728904846e-05, 0.0001778233735145764, 0.0006138326066512781, 0.0015764725998119176, 0.0037103530819868344, 0.0067931948362828076, 0.01079080106010088, 0.015043173463281182, 0.019138240574506284, 0.02257672907583141, 0.024568692827220656, 0.024678122595537318, 0.023575275711720953, 0.021655125245789518, 0.018941608959562282, 0.015374882448491067, 0.012567324955116697, 0.009776865863041806, 0.007282209113447893, 0.005380866888945884, 0.0035855347525006413, 0.00184491750021373, 0.0006394802086004958, 0.00014020689065572369, 2.3937761819269897e-05, 1.7098401299478499e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
p_chilist = [0.02612190922752951, 0.03813133839809607, 0.05274537644458024, 0.06489644319222346, 0.07288634782323605, 0.07407749922086473, 0.07299946665753995, 0.06875902995856667, 0.0645609562975759, 0.06011837615438984, 0.055248406766742184, 0.05103090160914082, 0.047756793488351365, 0.04413118181103418, 0.041053908781678826, 0.03832993275589676, 0.036825771483521286, 0.03598947857032338, 0.03549244781354302, 0.03496777046447948, 0.03071850960280496, 0.029766407119021135, 0.025033072148163225, 0.01653225806451613, 0.002544529262086514, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p_penglist = [0.016593743816215572, 0.017955302288611433, 0.018465788206656883, 0.01839723066178121, 0.018182044526884347, 0.01826182805677133, 0.01847633501601641, 0.018853696343716912, 0.0194061569906112, 0.019873991200097366, 0.02038180773636104, 0.020960620841050458, 0.02166036189591087, 0.022295544318193358, 0.02313574577564478, 0.02423336565601665, 0.025661475883795585, 0.02726837561692224, 0.02865446909849814, 0.029866280672732285, 0.03262217670819821, 0.03397112056963284, 0.030728557188055304, 0.030556356298645143, 0.024006001500375095, 0.037037037037037035, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p_ganglist = [0.0005133318937177928, 0.0007106881994146458, 0.0008778412764380955, 0.0010614442223694858, 0.0012789107971170144, 0.0015918069564673338, 0.0020650851501244328, 0.002590564515104147, 0.0033454503815608673, 0.004143050548380717, 0.00520405346977294, 0.006524799513432833, 0.008188547566572385, 0.009750104040744337, 0.012143771282309116, 0.014479542529172206, 0.01711312860794854, 0.021578495396720788, 0.024256515796584607, 0.029570170895908854, 0.037483898668956635, 0.03863011663323676, 0.044552896725440806, 0.04309949564419991, 0.023706896551724137, 0.018867924528301886, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p_tinghu = [0.009865718498528235, 0.6511627906976745, 0.6720357941834452, 0.6773132926256459, 0.6619363204149258, 0.649500384319754, 0.6175354989301692, 0.5921834418472267, 0.5549132947976878, 0.5242381177956734, 0.4987929528994812, 0.4706194174242625, 0.4467869718309859, 0.4198679113265184, 0.4033761391880696, 0.38783431700962406, 0.3675793469473832, 0.36305885750036093, 0.34632328024398507, 0.3290989660265879, 0.3170805572380376, 0.30323325635103926, 0.28665568369028005, 0.25, 0.23671497584541062, 0.2692307692307692, 0.6666666666666666, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
chu_net = Net(shape8.CHU_SHAPE, 0.01, 34, model_file=chu_net_file)
peng_net = Net(shape8.PENG_SHAPE, 0.01, shape8.PENG_OUT_NUM, model_file=peng_net_file)
chow_net = Net(shape8.CHOW_SHAPE, 0.01, shape8.CHOW_OUT_NUM, model_file=chow_net_file)
def hand2card(hand):
    card = []
    for i in range(36):
        if hand[i]:
            for ii in range(hand[i]):
                card.append(i)
    return card
def mopai_chupai(remain):
    num = 0
    for i in range(36):
        num += remain[i]
    if num == 0:
        # print('num',num)
        return None
    c = random.randint(0,num-1)
    for i in range(36):
        c -= remain[i]
        if c<0:
            return i
def _get_peng_decision(ta_data):
    t0 = datetime.now()
    peng_data = np.array(ta_data)
    peng_data = peng_data.reshape(1, shape8.PENG_SHAPE[0], shape8.PENG_SHAPE[1], shape8.PENG_SHAPE[2])
    result = peng_net.policy_value(peng_data)
    #print('PENG: ',result)
    #exit()
    sorted_values = np.argsort(-result[0])
    decison = sorted_values[0]
    t1 = datetime.now()
    #self.time_get_peng.append((t1-t0).total_seconds())
    if decison:
        return True
    else:
        return False
def get_ta_action(my_seat, recorder, quanfeng):
    # print('recorder',recorder)
    '''
    13张手牌的情况,无明杠逻辑
    :param my_seat:
    :param recorder:
    :param quanfeng:
    :return:
    '''
    #players, is_gang, wall_left, table, target_card, lst_seat = self.restore_player_etc(my_seat, recorder)

    t0 = datetime.now()
    wall_left, table, lsts, their_walls, players,shows_1,outs_1 = restore_player_etc(my_seat, recorder)

    lst1_his = lsts[-1]
    lst2_his = lsts[-2]

    if lst2_his[0] == my_seat and lst2_his[1] in [ACTION_TYPE.ANGANG, ACTION_TYPE.BUGANG, ACTION_TYPE.MINGGANG]:
        is_gang = True
    else:
        is_gang = False
    target_card = lst1_his[2]
    lst_seat = lst1_his[0]

    is_zi = False
    hand = players[my_seat].hand
    pack = players[my_seat].show_packs
    # shows = {i:players[i].show for i in players}
    menFeng = my_seat
    shows = {i:shows_1[i] for i in range(4)}
    outs = {i:outs_1[i] for i in range(4)}
    quanFeng = quanfeng
    '''
    if wall_left > 0:
        is_last = False
    else:
        is_last = True
    '''

    is_last = False
    if_hu, fan, ans = PublicShareFun.botzone_fan_cal(is_zi, is_gang, is_last, table, hand,pack,shows, target_card, menFeng, quanFeng)

    seat = my_seat
    action = None
    card = target_card

    if if_hu:
        action = ACTION_TYPE.HU
    else:
        player_cards = {i:players[i].hand for i in players}
        # outs = {i:players[i].show for i in players}
        remains = {i:players[i].remain for i in players}
        packs = {i:players[i].show_packs for i in players}
        t2 = datetime.now()
        ta_data = PublicShareFun.make_ta_data_alpha(player_cards,shows,outs,remains, packs,target_card, my_seat, quanfeng, wall_left, table, lst1_his, lst2_his)
        #TODO
        #ting_state = PublicShareFun.if_ting(player_cards,shows,outs,remains, packs,target_card, my_seat, quanfeng, wall_left, table, lst1_his, lst2_his,if_zi_state=False)
        t3 = datetime.now()
        if_peng = False
        if players[my_seat].hand[target_card] == 2:
            if_peng = _get_peng_decision(ta_data)

        if_chow = False
        could_chow, chow_types = PublicShareFun.judge_chow(lst_seat, my_seat, target_card, players[my_seat].hand)
        if could_chow:
            if_chow = _get_chow_decision(ta_data)

        if if_peng and not if_chow:
            action = ACTION_TYPE.PENG
            card = target_card
            #return my_seat, PENG, target_card, None
        elif if_chow and not if_peng:
            chow_type = chow_types[0] #TODO 有多个吃类型时，选第一个
            action = chow_type
            #return my_seat, chow_type, target_card, None
        elif not if_peng and not if_chow:
            action = ACTION_TYPE.PASS
            #return my_seat, PASS, target_card, None
        else:
            action = ACTION_TYPE.PENG
            #return my_seat, PENG, target_card, None #TODO 可以碰吃的时候如何选择

    t1 = datetime.now()
    #self.time_ta.append((t1-t0).total_seconds())
    return seat, action, card, fan
def _get_chow_decision(ta_data):
    t0 = datetime.now()
    ta_data = np.array(ta_data)
    chow_data = ta_data.reshape(1, shape8.CHOW_SHAPE[0], shape8.CHOW_SHAPE[1], shape8.CHOW_SHAPE[2])
    result = chow_net.policy_value(chow_data)

    sorted_values = np.argsort(-result[0])

    decision = sorted_values[0]

    t1 = datetime.now()
    #self.time_get_chow.append((t1-t0).total_seconds())
    if decision:
        return True
    else:
        return False

def _get_chu_card(chu_data,his_hand):
    t0 = datetime.now()
    chu_data = np.array(chu_data)
    chu_data = chu_data.reshape(1, shape8.CHU_SHAPE[0], shape8.CHU_SHAPE[1], shape8.CHU_SHAPE[2])

    #rw.write_to_pickle(chu_data, 'chu_data.pickle')
    result = chu_net.policy_value(chu_data) # sum(result) = 0
    # print('result',result)

    sorted_values = np.argsort(-result[0])
    for v in sorted_values:
        if his_hand[v] > 0:
            # print(his_hand,v)
            index = v
            break
    t1 = datetime.now()
    #self.time_get_chu.append((t1-t0).total_seconds())
    # print('index',index)
    return index

def canpeng(remain,cardvalue,nowseat,chulist):
    if remain[cardvalue] >= 2:
        a = chulist[nowseat]
        b = random.random()
        if a >=40:
            return False
        c = p_penglist[a]
        if b<=c:
            return True
        else:
            return False
    return False

def canchi(remain,cardvalue,nowseat,chulist):
    zuochi = 0
    zhongchi = 0
    youchi = 0
    if cardvalue<27:
        if cardvalue <= 24 and remain[cardvalue+1] >0 and remain[cardvalue+2]>0 and int((cardvalue+2)/9) == int(cardvalue/9):
            zuochi = 1
        if cardvalue <= 25 and cardvalue>=1 and remain[cardvalue+1] >0 and remain[cardvalue-1]>0 and int((cardvalue+1)/9) == int((cardvalue-1)/9):
            zhongchi = 1
        if cardvalue >= 2 and remain[cardvalue-1] >0 and remain[cardvalue-2]>0 and int((cardvalue-2)/9) == int(cardvalue/9):
            youchi = 1

    if zuochi == 1 or zhongchi == 1 or youchi == 1:
        a = chulist[nowseat]
        if a >= 40:
            return False,None
        b = random.random()
        c = p_chilist[a]
        if b<=c:
            cishixuanze = []
            if zuochi:
                cishixuanze.append(0)
            if zhongchi:
                cishixuanze.append(1)
            if youchi:
                cishixuanze.append(2)
            d = random.randint(0,len(cishixuanze)-1)

            return True,cishixuanze[d]
        else:
            return False,None
    return False,None

def cangang(remain,cardvalue,nowseat,chulist):
    if remain[cardvalue] > 2:
        a = chulist[nowseat]
        if a >=40:
            return False
        b = random.random()
        c = p_ganglist[a]
        if b<=c:
            return True
        else:
            return False
    return False

def canhu(nowseat,chulist):
    a = chulist[nowseat]

    if a>=40:
        # print(chulist)
        b = 0
    else:
        b = p_hulist[a]
    c = random.random()
    if c <= b:
        return True
    else:
        return False

def get_zi_action(my_seat, recorder, quanfeng):
    # print('recorder',recorder)

    #players, is_gang, wall_left, table, target_card, lst_seat = self.restore_player_etc(my_seat, recorder)
    #使用wall_left参数
    # print(recorder)
    t0 = datetime.now()
    wall_left, table, lsts, their_walls, players,shows_1,outs_1 = restore_player_etc(my_seat, recorder)
    # print('wall_left, table, lsts, their_walls, players', wall_left, table, lsts, their_walls, players)
    lst1_his = lsts[-1]
    lst2_his = lsts[-2]

    if lst2_his[0] == my_seat and lst2_his[1] in [ACTION_TYPE.ANGANG, ACTION_TYPE.BUGANG, ACTION_TYPE.MINGGANG]:
        is_gang = True
    else:
        is_gang = False
    target_card = lst1_his[2]

    #自摸胡， 出牌， 若出牌恰巧可以 暗杠、补杠， 则暗杠或补杠
    is_zi = True

    hand = players[my_seat].hand
    pack = players[my_seat].show_packs
    # shows ={i:players[i].show for i in players}
    shows = {i:shows_1[i] for i in range(4)}
    outs = {i:outs_1[i] for i in range(4)}
    menFeng = my_seat
    quanFeng = quanfeng
    #is_zi, is_gang, is_last, table, hand,pack,shows, target_card, menFeng, quanFeng
    '''
    if wall_left > 0:
        is_last = False
    else:
        is_last = True
    '''


    is_last = False

    if_hu, fan, ans = PublicShareFun.botzone_fan_cal(is_zi, is_gang, is_last, table, hand,pack,shows, target_card, menFeng, quanFeng)

    my_seat = my_seat
    action = None
    card = None
    #fan = None
    if if_hu:
        action = ACTION_TYPE.HU
        card = target_card

        #return my_seat, HU, target_card, fan

    else:
        for i in players:
            if i == menFeng:
                player_cards = {i: players[i].hand}
                # outs = {i: players[i].show}
                remains = {i: players[i].remain}
                packs = {i: players[i].show_packs}
        # player_cards,shows,outs,remains, packs,target_card, menfeng, quanfeng, wall_left, table, lst1_his, lst2_his
        t2 = datetime.now()

        chu_data = PublicShareFun.make_zi_data_alpha(player_cards, shows, outs, remains, packs, target_card, my_seat,
                                                      quanfeng, wall_left, table, lst1_his, lst2_his)
        t3 = datetime.now()

        chu_card = _get_chu_card(chu_data, players[0].hand)
        # print('players[0].hand',players[0].hand)
        card = chu_card
        if players[my_seat].hand[chu_card] == 4:
            action = ACTION_TYPE.ANGANG
            card = chu_card
        # print('ziactionkaitou')
        # elif players[my_seat].could_bugang(chu_card):
        #     action = ACTION_TYPE.BUGANG
        #     card = chu_card

    t1 = datetime.now()
    # self.time_zi.append((t1-t0).total_seconds())
    # print('ziactionjieshu')
    return my_seat, action, card, players[0].hand

def restore_player_etc(my_seat, recorder):
    t0 = datetime.now()

    players = {0:Player('GBMJ')}

    quanfeng = recorder[0][0]

    initial = recorder[1]

    if my_seat == 0:
        seat = initial[0]
        hand = initial[1]
        flower = initial[2]

        for c in hand:
            players[seat].draw_card(True, c)

        for c in flower:
            players[seat].draw_card(True, c)
            players[seat].compensate_flower(True, c)
    my_history = recorder[2:]
    wall_left, table, lsts, their_walls, players,shows,outs = on_my_history(my_history, my_seat, players)


    t1 = datetime.now()
    #self.time_history.append((t1-t0).total_seconds())
    return wall_left, table, lsts, their_walls, players,shows,outs

def on_my_history(my_history, my_seat, players):
    #TODO 复试赛制，则使用their_walls
    shows = np.zeros((4,36),dtype = np.int)
    outs = np.zeros((4,36),dtype = np.int)

    table = [0] * 36 #todo
    their_walls = {} #记录每人的牌墙

    wall_left = 4*36 - 13*4
    for p in players:
        their_walls[p] = 36 #初始每人36张
        their_walls[p] -= 13 #初始手牌

        show = players[p].show
        for s in show:
            wall_left -= s
            their_walls[p] -= s

    for i in range(len(my_history)):
        item = my_history[i]
        action_pos = item[0]
        action_type = item[1]
        card_value = item[2]

        if my_seat == action_pos:
            if_self = True
        else:
            if_self = False

        if action_type in [ACTION_TYPE.ZHUAPAI, ACTION_TYPE.ZHUAGANG, ACTION_TYPE.ZHUABU]:
            if action_pos == 0:
                # print('dnasjdiausudn',players[action_pos].remain)
                players[action_pos].draw_card(if_self, card_value)
                # print('jdhaishbdiasbd',players[action_pos].remain)
                wall_left -= 1
                their_walls[action_pos] -= 1
        elif action_type == ACTION_TYPE.BUHUA:
            if action_pos == 0:
                players[action_pos].compensate_flower(if_self, card_value)
            else:
                players[0].remain[card_value] -= 1

        elif action_type in [ACTION_TYPE.SHOUQIE, ACTION_TYPE.MOQIE]:
            table[card_value] += 1
            outs[action_pos][card_value] += 1
            if action_pos == 0:
                players[action_pos].discard_card(if_self, card_value)

            else:
                players[0].reduce_remain(if_self, card_value)

        elif action_type == ACTION_TYPE.PENG:
            shows[action_pos][card_value] += 3
            table[card_value] -= 1
            lst_seat = my_history[i - 1][0]
            if action_pos == 0:
                players[action_pos].peng(if_self, card_value, lst_seat, action_pos)
            else:

                for _ in range(2):
                    players[0].reduce_remain(if_self, card_value)
        elif action_type == ACTION_TYPE.BUGANG:
            if action_pos ==0:
                players[action_pos].bugang(if_self, card_value)
            else:
                players[0].reduce_remain(if_self, card_value)

        elif action_type == ACTION_TYPE.ANGANG:
            if action_pos == 0:
                players[action_pos].angang(if_self, card_value, action_pos)
            '''
            for p in players:
                if p != action_pos:
                    for _ in range(4):
                        players[p].reduce_remain(if_self, card_value)
            '''
        elif action_type == ACTION_TYPE.MINGGANG:
            shows[action_pos][card_value] += 4
            table[card_value] -= 1
            lst_seat = my_history[i - 1][0]
            if action_pos == 0:
                # print('明杠',if_self, card_value, lst_seat, action_pos)
                players[action_pos].minggang(if_self, card_value, lst_seat, action_pos)
            else:
                for _ in range(3):
                    players[0].reduce_remain(if_self, card_value)
        elif action_type in [ACTION_TYPE.CHOW_LEFT, ACTION_TYPE.CHOW_MIDDLE, ACTION_TYPE.CHOW_RIGHT]:

            table[card_value] -= 1
            if action_pos == 0:
                players[action_pos].chow(if_self, card_value, action_type)
            if action_type == ACTION_TYPE.CHOW_LEFT:
                shows[action_pos][card_value]+=1
                shows[action_pos][card_value+1]+=1
                shows[action_pos][card_value+2]+=1
                cards = [card_value, card_value + 1, card_value + 2]
            elif action_type == ACTION_TYPE.CHOW_MIDDLE:
                shows[action_pos][card_value] += 1
                shows[action_pos][card_value + 1] += 1
                shows[action_pos][card_value -1] += 1
                cards = [card_value - 1, card_value, card_value + 1]
            else:
                shows[action_pos][card_value] += 1
                shows[action_pos][card_value - 1] += 1
                shows[action_pos][card_value - 2] += 1
                cards = [card_value - 2, card_value - 1, card_value]
            if action_pos != 0:
                for c in cards:
                    if c != card_value:
                        players[0].reduce_remain(if_self, c)

    lsts = []
    reversed_history = list(reversed(my_history))
    if my_history[-1][0] == my_seat:

        for i in range(len(reversed_history)):
            if reversed_history[i][0] != my_seat:
                break
            if reversed_history[i][1] == ACTION_TYPE.BUHUA:
                pass
            else:
                lsts.append(reversed_history[i])
            if len(lsts) == 2:
                break
    else:
        lsts.append(reversed_history[0])

    lsts = list(reversed(lsts))
    if len(lsts) < 2:
        lsts.insert(0,[None, None, None])
    #for i in range(len(my_history)): print(i, '   ', my_history[i]  )

    #返回与botzone接口不同， 多返回一个players
    players[0].hand[34] = 0
    players[0].hand[35] = 0
    return wall_left, table, lsts, their_walls, players, shows, outs

def get_data(my_seat, recorder, quanfeng):


    #players, is_gang, wall_left, table, target_card, lst_seat = self.restore_player_etc(my_seat, recorder)
    #使用wall_left参数
    # print(recorder)
    t0 = datetime.now()
    t_0 = time.time()
    wall_left, table, lsts, their_walls, players,shows_1,outs_1 = restore_player_etc(my_seat, recorder)
    # print('1111',time.time() - t_0)
    # print('wall_left, table, lsts, their_walls, players', wall_left, table, lsts, their_walls, players)
    lst1_his = lsts[-1]
    lst2_his = lsts[-2]

    if lst2_his[0] == my_seat and lst2_his[1] in [ACTION_TYPE.ANGANG, ACTION_TYPE.BUGANG, ACTION_TYPE.MINGGANG]:
        is_gang = True
    else:
        is_gang = False
    target_card = lst1_his[2]

    #自摸胡， 出牌， 若出牌恰巧可以 暗杠、补杠， 则暗杠或补杠
    is_zi = True

    hand = players[my_seat].hand
    pack = players[my_seat].show_packs
    # shows ={i:players[i].show for i in players}
    shows = {i:shows_1[i] for i in range(4)}
    outs = {i:outs_1[i] for i in range(4)}

    menFeng = my_seat
    quanFeng = quanfeng
    #is_zi, is_gang, is_last, table, hand,pack,shows, target_card, menFeng, quanFeng
    '''
    if wall_left > 0:
        is_last = False
    else:
        is_last = True
    '''


    is_last = False

    if_hu, fan, ans = PublicShareFun.botzone_fan_cal(is_zi, is_gang, is_last, table, hand,pack,shows, target_card, menFeng, quanFeng)

    my_seat = my_seat
    action = None
    card = None
    #fan = None
    if if_hu:
        action = ACTION_TYPE.HU
        card = target_card

        #return my_seat, HU, target_card, fan

    else:
        for i in players:
            if i == menFeng:
                player_cards = {i: players[i].hand}
                # outs = {i: players[i].show}
                remains = {i: players[i].remain}
                packs = {i: players[i].show_packs}
        # player_cards,shows,outs,remains, packs,target_card, menfeng, quanfeng, wall_left, table, lst1_his, lst2_his
        t2 = datetime.now()
        t_3 = time.time()
        chu_data = PublicShareFun.make_zi_data_alpha(player_cards, shows, outs, remains, packs, target_card, my_seat,
                                                      quanfeng, wall_left, table, lst1_his, lst2_his)
        t3 = datetime.now()
        # print('新时间',time.time() - t_3)
        chu_card = _get_chu_card(chu_data, players[0].hand)
        # print('players[0].hand',players[0].hand)
        card = chu_card
        if players[my_seat].hand[chu_card] == 4:
            action = ACTION_TYPE.ANGANG
            card = chu_card
        # print('ziactionkaitou')
        # elif players[my_seat].could_bugang(chu_card):
        #     action = ACTION_TYPE.BUGANG
        #     card = chu_card

    t1 = datetime.now()
    # print('2',time.time() - t_0)
    # self.time_zi.append((t1-t0).total_seconds())
    # print('ziactionjieshu')
    return players[0].hand,players[0].remain,players[0].show,players[0].out

def get_chulist(recorder):
    chulist = [0]*4
    recorder_n = recorder[2:]
    for i in range(len(recorder_n)):
        if recorder_n[i][1] == 2 or recorder_n[i][1] == 3 or recorder_n[i][1] == 14:
            chulist[recorder_n[i][0]] += 1
    return chulist

def MCTS(recorder,nowseat,discard):
    hand, remain, shows, _ = get_data(0,recorder,0)
    chulist = get_chulist(recorder)
    NOWSTATE = 'CHUSTATE'
    ziwozuoci = nowseat
    # print('ziwozuoci',ziwozuoci)
    NOW_SEAT = nowseat
    GAME_OVER = False
    FIRST = True
    while not GAME_OVER:
        # print('recorder', recorder)
        # print('remain', remain)
        # print('hand',hand)
        # a,b,c,d = get_data(0,recorder,0)
        # for i in range(34):
        #     if remain[i] != b[i]:
        #         print('这里不一样这里不一样',b)
        if NOWSTATE == 'ZISTATE':
            # print('ZISTATE')
            # print('NOW_SEAT',NOW_SEAT)
            cardvalue = mopai_chupai(remain)
            # print('我摸',cardvalue)

            if cardvalue == None:
                GAME_OVER = True
                return -1

            while cardvalue > 33:
                action_seat0, action_type0, card_value0, the_changed_score = NOW_SEAT, ACTION_TYPE.BUHUA, cardvalue, None
                recorder.append([action_seat0, action_type0, card_value0, 1])

                remain[cardvalue] -= 1
                hand[cardvalue] += 1
                # print(NOW_SEAT, '摸了', cardvalue)
                # print(NOW_SEAT, '的手牌', hand)
                hand[cardvalue] -= 1
                # print(NOW_SEAT, '出了', cardvalue)
                # print(NOW_SEAT, '的手牌', hand)
                cardvalue = mopai_chupai(remain)
                if cardvalue == None:
                    GAME_OVER = True
                    return -1
                # remain[cardvalue] -= 1
                # hand[cardvalue] += 1
                # recorder.append([action_seat0, action_type0, card_value0, the_changed_score])
            remain[cardvalue] -= 1
            hand[cardvalue] += 1
            # print(NOW_SEAT,'摸了',cardvalue)
            # print(NOW_SEAT,'的手牌',hand)
            if cardvalue is not None:
                recorder.append([NOW_SEAT, ACTION_TYPE.ZHUAPAI, int(cardvalue), 1])
            else:
                recorder.append([NOW_SEAT, None, cardvalue, 1])
            #todo 判断自己的动作根据网络

            action_seat, action_type, cardvalue, fan = get_zi_action(NOW_SEAT,recorder,quanfeng=0)
            # if fan != hand:
                # print('看这看这看这')

            if action_type == ACTION_TYPE.HU:
                # print('我胡')
                recorder.append([action_seat, action_type, cardvalue, 1, fan])
                # print('自摸')
                return 0
            elif action_type == ACTION_TYPE.ANGANG:
                # print('我暗杠')
                hand[cardvalue] -= 4
                recorder.append([NOW_SEAT, ACTION_TYPE.ANGANG, int(cardvalue), 1])
                shows[cardvalue] += 4
                NOW_SEAT = NOW_SEAT  # 暗杠后自己抓牌
                NOWSTATE = 'ZISTATE'
            elif action_type == ACTION_TYPE.BUGANG:
                # print('我补杠')
                hand[cardvalue] -= 1
                recorder.append([NOW_SEAT, ACTION_TYPE.BUGANG, int(cardvalue), 1])
                shows[cardvalue] += 1
                NOWSTATE = 'ZISTATE'
                #抢杠胡没做
            elif action_type == None:
                NOW_SEAT = NOW_SEAT
                NOWSTATE = 'CHUSTATE'
            # print(action_type,cardvalue)

        if NOWSTATE == 'CHUSTATE':
            action_seat, action_type, cardvalue, fan = get_zi_action(NOW_SEAT,recorder,quanfeng=0)
            # print('hand',hand)
            # print('CHUSTATE')
            # print('我出',cardvalue)
            if FIRST:
                if hand[discard]:
                    cardvalue = discard
                    FIRST = False
            # print('cardvalue',cardvalue)
            #TODO 根据已有状态判断出的牌是哪张
            recorder.append([NOW_SEAT, ACTION_TYPE.SHOUQIE, int(cardvalue), 1])
            hand[cardvalue] -= 1
            chulist[NOW_SEAT] += 1
            # print(NOW_SEAT,'出了',cardvalue)
            # print(NOW_SEAT,'的手牌',hand)
            # seat1_peng = random.random()
            # seat1_chi = random.random()
            hu1 = canhu((NOW_SEAT+1)%4,chulist)
            hu2 = canhu((NOW_SEAT+2)%4,chulist)
            hu3 = canhu((NOW_SEAT+3)%4,chulist)
            peng = canpeng(remain,cardvalue,NOW_SEAT,chulist)
            gang = cangang(remain,cardvalue,NOW_SEAT,chulist)
            chi,chi_type = canchi(remain,cardvalue,NOW_SEAT,chulist)
            # print('22222222222222')
            if hu1:
                # print('1hu')
                return 1
            if hu2:
                # print('2hu')
                return 2
            if hu3:
                # print('3hu')
                return 3
            elif gang:
                remain[cardvalue] -= 3
                gang_person = random.randint(1,3)
                recorder.append([(NOW_SEAT+gang_person)%4, ACTION_TYPE.MINGGANG, int(cardvalue), 1])

                NOW_SEAT = (NOW_SEAT+gang_person)%4
                # print(NOW_SEAT,'杠')
                NOWSTATE = 'JIQIMOSTATE'
            elif peng:
                remain[cardvalue] -= 2
                peng_person = random.randint(1,3)
                recorder.append([(NOW_SEAT+peng_person)%4, ACTION_TYPE.PENG, int(cardvalue), 1])
                NOW_SEAT = (NOW_SEAT+peng_person)%4
                # print(NOW_SEAT,'碰')
                NOWSTATE = 'JIQICHUSTATE'
            elif chi:
                if chi_type == 0:
                    remain[cardvalue+1] -= 1
                    remain[cardvalue+2] -= 1
                    recorder.append([(NOW_SEAT+1)%4, ACTION_TYPE.CHOW_LEFT, int(cardvalue), 1])
                elif chi_type == 1:
                    remain[cardvalue + 1] -= 1
                    remain[cardvalue - 1] -= 1
                    recorder.append([(NOW_SEAT+1)%4, ACTION_TYPE.CHOW_MIDDLE, int(cardvalue), 1])
                elif chi_type == 2:
                    remain[cardvalue - 1] -= 1
                    remain[cardvalue - 2] -= 1
                    recorder.append([(NOW_SEAT+1)%4, ACTION_TYPE.CHOW_RIGHT, int(cardvalue), 1])
                NOW_SEAT = (NOW_SEAT+1)%4
                # print(NOW_SEAT,'吃')
                NOWSTATE = 'JIQICHUSTATE'

            else:

                NOW_SEAT = (NOW_SEAT+1)%4
                NOWSTATE = 'JIQIMOSTATE'

        if NOWSTATE == 'JIQIMOSTATE':
            # print('机器摸牌')
            # action_seat, action_type, cardvalue, fan = get_zi_action(0,recorder,quanfeng=0)
            # print('hand',hand)
            # print('现在座位',NOW_SEAT)
            shengyunum = 0
            for i in range(36):
                shengyunum += remain[i]
            # print(shengyunum,chulist)
            # print('JIQIMOPAI')
            # print('NOW_SEAT',NOW_SEAT)
            #todo 机器补花的概率
            hu = canhu(NOW_SEAT,chulist)
            # gang_1 = cangang(remain)#todo 暗杠概率没算
            # bugang = canbugang(lunshu)
            if hu:
                # print(NOW_SEAT,'胡')
                return NOW_SEAT
            # elif gang_1:
            #     NOWSTATE = 'JIQIMOSTATE'
            # elif bugang:
            #     NOWSTATE = 'JIQIMOSTATE'

            else:
                NOWSTATE = 'JIQICHUSTATE'

        if NOWSTATE == 'JIQICHUSTATE':
            # print('现在座位',NOW_SEAT)
            # print('han',hand)
            # action_seat, action_type, cardvalue, fan = get_zi_action(0,recorder,quanfeng=0)
            # print('JIQICHUSTATE')
            # print('NOW_SEAT',NOW_SEAT)
            change = False
            cardvalue = mopai_chupai(remain)
            # print('机器出的cardvalue',cardvalue)
            # print('type(cardvalue)',type(cardvalue))
            # print('==',cardvalue == None)
            # print('is',cardvalue is None)
            if cardvalue == None:
                return -1
            # print('机器出的cardvalue',cardvalue)
            # print('type(cardvalue)',type(cardvalue))
            while cardvalue > 33:  # todo 补花
                recorder.append([NOW_SEAT, ACTION_TYPE.SHOUQIE, int(cardvalue), 1])
                remain[cardvalue] -= 1
                chulist[NOW_SEAT] += 1
                # print(NOW_SEAT, '补花', cardvalue)
                # print(NOW_SEAT, '的手牌', hand)
                cardvalue = mopai_chupai(remain)
                if cardvalue == None:
                    return -1
            recorder.append([NOW_SEAT, ACTION_TYPE.SHOUQIE, int(cardvalue), 1])
            remain[cardvalue] -= 1
            chulist[NOW_SEAT] += 1
            # print(NOW_SEAT,'出了',cardvalue)
            # print('我的remain',remain)

            _, action_panduan,cardvalue,fan = get_ta_action(ziwozuoci,recorder,0)
            # print('我的动作',action_panduan)
            for i in range(4):
                if i!=ziwozuoci and i!=NOW_SEAT:
                    aa = canhu(i, chulist)
                    aa_1 = random.random()
                    if aa_1<aa:
                        # print('机器点机器胡')
                        return i
            if action_panduan == ACTION_TYPE.HU:
                # print(cardvalue)
                # print('他点炮我胡')
                return 0
            # print('cardvalue',cardvalue)
            aa = cangang(remain,cardvalue,NOW_SEAT, chulist)
            aa_1 = random.random()
            # print('daskdnoasjdnjasd')
            if aa_1 < aa:
                change = True
                c=[]
                for i in range(4):
                    if (ziwozuoci+i) != NOW_SEAT and ((ziwozuoci+i))%4 != ziwozuoci:
                        c.append((ziwozuoci+i)%4)
                d = random.sample(c,1)[0]
                remain[cardvalue] -= 3
                recorder.append([d, ACTION_TYPE.MINGGANG, int(cardvalue), 1])
                NOW_SEAT = d
                # print(NOW_SEAT,'明杠')
                # print('1',NOW_SEAT)
                NOWSTATE = 'JIQIMOSTATE'
            if hand[cardvalue] == 3:
                recorder.append([ziwozuoci, ACTION_TYPE.MINGGANG, int(cardvalue), 1])
                change = True
                NOW_SEAT = ziwozuoci
                hand[cardvalue] -= 3
                # print(NOW_SEAT,'明杠')
                NOWSTATE = 'ZISTATE'
            if change ==False:
                aa = canpeng(remain,cardvalue,NOW_SEAT, chulist)
                aa_1 = random.random()
                if aa_1<aa:
                    change = True
                    c=[]
                    for i in range(4):
                        if (ziwozuoci + i) != NOW_SEAT and ((ziwozuoci + i)) % 4 != ziwozuoci:
                            c.append((ziwozuoci+i)%4)
                    d = random.sample(c,1)[0]
                    remain[cardvalue] -= 2
                    recorder.append([d, ACTION_TYPE.PENG, int(cardvalue), 1])
                    NOW_SEAT = d
                    # print(NOW_SEAT,'碰')
                    # print('2', NOW_SEAT)
                    NOWSTATE = 'JIQIMOSTATE'
            #判断自己碰
            # print('njdkasdnkjsandja')
            if change == False:
                if hand[cardvalue] == 2 and action_panduan == ACTION_TYPE.PENG:
                    recorder.append([ziwozuoci, ACTION_TYPE.PENG, int(cardvalue), 1])
                    change = True
                    # print('这里碰了')
                    # print('我碰')
                    # print('碰前',hand)
                    hand[cardvalue] -= 2
                    # print('碰后',hand)
                    NOW_SEAT = ziwozuoci
                    # print('我碰',ziwozuoci)
                    # print('碰之后的座位',NOW_SEAT)
                    # print('cardvalue',cardvalue)
                    # print('3', NOW_SEAT)
                    NOWSTATE = 'CHUSTATE'

            if change == False and (NOW_SEAT+1)%4 != ziwozuoci:
                # print(remain,cardvalue,NOW_SEAT, chulist)
                aa,bbb = canchi(remain,cardvalue,NOW_SEAT, chulist)
                aa_1 = random.random()
                if aa_1<aa:
                    change = True
                    d = NOW_SEAT + 1

                    if bbb == 0:
                        remain[cardvalue+1] -= 1
                        remain[cardvalue + 2] -= 1
                        recorder.append([d, ACTION_TYPE.CHOW_LEFT, int(cardvalue), 1])
                        NOW_SEAT = d
                        # print(NOW_SEAT,'左吃')
                        # print('4', NOW_SEAT)
                        NOWSTATE = 'JIQIMOSTATE'
                    if bbb == 1:
                        remain[cardvalue-1] -= 1
                        remain[cardvalue + 1] -= 1
                        recorder.append([d, ACTION_TYPE.CHOW_MIDDLE, int(cardvalue), 1])
                        NOW_SEAT = d
                        # print(NOW_SEAT,'中吃')
                        # print('5', NOW_SEAT)
                        NOWSTATE = 'JIQIMOSTATE'
                    if bbb == 2:
                        remain[cardvalue-1] -= 1
                        remain[cardvalue - 2] -= 1
                        recorder.append([d, ACTION_TYPE.CHOW_RIGHT, int(cardvalue), 1])
                        NOW_SEAT = d
                        # print(NOW_SEAT,'右吃')
                        # print('6', NOW_SEAT)
                        NOWSTATE = 'JIQIMOSTATE'
            # print('djasbdnabsdkhabsdasd')
            if change ==False and (NOW_SEAT+1)%4 == ziwozuoci:
                leftchi = rightchi = middlechi = False
                if cardvalue<25:
                    if hand[cardvalue+1] and hand[cardvalue+2] and int(cardvalue/9) == int((cardvalue+2)/9):
                        leftchi = True
                if cardvalue>2:
                    if hand[cardvalue-1] and hand[cardvalue-2] and int(cardvalue/9) == int((cardvalue-2)/9):
                        rightchi = True
                if cardvalue<26 and cardvalue>1:
                    if hand[cardvalue+1] and hand[cardvalue-1] and int((cardvalue+1)/9) == int((cardvalue-1)/9):
                        middlechi = True
                if middlechi or leftchi or rightchi:
                    if action_panduan == ACTION_TYPE.CHOW or action_panduan == ACTION_TYPE.CHOW_LEFT or action_panduan == ACTION_TYPE.CHOW_MIDDLE or action_panduan == ACTION_TYPE.CHOW_RIGHT:
                        # chikongjian = []
                        change = True
                        # if action_panduan == ACTION_TYPE.:
                        #     chikongjian.append(1)
                        # if leftchi:
                        #     chikongjian.append(0)
                        # if rightchi:
                        #     chikongjian.append(2)
                        # xuanze = random.sample(chikongjian,1)[0]
                        if action_panduan == ACTION_TYPE.CHOW_LEFT:
                            hand[cardvalue+1] -= 1
                            hand[cardvalue+2] -= 1
                            recorder.append([ziwozuoci, ACTION_TYPE.CHOW_LEFT, int(cardvalue), 1])
                            NOW_SEAT = ziwozuoci
                            # print(NOW_SEAT, '我左吃')
                            # print(hand)
                            # print('7', NOW_SEAT)
                            NOWSTATE = 'CHUSTATE'
                        if action_panduan == ACTION_TYPE.CHOW_MIDDLE:
                            hand[cardvalue+1] -= 1
                            hand[cardvalue-1] -= 1
                            recorder.append([ziwozuoci, ACTION_TYPE.CHOW_MIDDLE, int(cardvalue), 1])
                            NOW_SEAT = ziwozuoci
                            # print(NOW_SEAT, '我中吃')
                            # print(hand)
                            # print('8', NOW_SEAT)
                            NOWSTATE = 'CHUSTATE'
                        if action_panduan == ACTION_TYPE.CHOW_RIGHT:
                                hand[cardvalue-1] -= 1
                                hand[cardvalue-2] -= 1
                                recorder.append([ziwozuoci, ACTION_TYPE.CHOW_RIGHT, int(cardvalue), 1])
                                NOW_SEAT = ziwozuoci
                                # print(NOW_SEAT, '我右吃')
                                # print(han)
                                # print('9', NOW_SEAT)
                                NOWSTATE = 'CHUSTATE'



                #依次看自己下三家胡牌，中间包含自己，碰杠牌，下家吃牌
            if change == False:
                # print('asndiuabsiudbans',NOW_SEAT)
                # print('NOW_SEAT',NOW_SEAT)
                NOW_SEAT = (NOW_SEAT+1)%4
                # print('nowseaDASDADASDAt',NOW_SEAT)
                if NOW_SEAT == ziwozuoci:
                    NOWSTATE = 'ZISTATE'
                else:
                    NOWSTATE = 'JIQIMOSTATE'


if __name__ == '__main__':
    time_now = time.time()
    recorder = [[0], [0, [0, 1, 2, 2, 7, 8, 8, 11, 13, 17, 18, 19, 22], []], [0, 1, 4, 9.0], [0, 2, 22, 1], [1, 2, 18, 1], [2, 2, 1, 1], [3, 2, 18, 1], [0, 1, 0, 1], [0, 2, 17, 1], [1, 2, 10, 1], [2, 2, 24, 1], [3, 2, 1, 1], [0, 1, 28, 1], [0, 2, 28, 1], [1, 2, 32, 1], [2, 2, 35, 1], [2, 2, 0, 1], [0, 5, 0, 1], [0, 2, 7, 1], [1, 2, 5, 1], [2, 2, 7, 1], [3, 2, 27, 1], [0, 1, 24, 1], [0, 2, 24, 1], [1, 2, 21, 1], [2, 2, 1, 1], [3, 40, 1, 1], [3, 2, 12, 1], [0, 41, 12, 1], [0, 2, 4, 1], [1, 2, 16, 1], [2, 42, 16, 1], [2, 2, 28, 1], [3, 2, 32, 1], [0, 1, 24, 1], [0, 2, 24, 1], [1, 2, 32, 1], [2, 2, 23, 1], [3, 41, 23, 1], [3, 2, 25, 1], [0, 1, 17, 1], [0, 2, 17, 1], [1, 2, 6, 1], [2, 2, 8, 1], [3, 2, 14, 1], [0, 1, 7, 1], [0, 2, 2, 1], [1, 40, 2, 1], [1, 2, 15, 1], [2, 2, 27, 1], [3, 2, 23, 1], [0, 1, 33, 1], [0, 2, 7, 1], [1, 2, 3, 1], [2, 2, 19, 1], [3, 5, 19, 1], [3, 2, 8, 1], [0, 1, 23, 1], [0, 2, 23, 1], [1, 2, 16, 1], [2, 2, 33, 1], [3, 2, 26, 1], [0, 1, 20, 1], [0, 2, 33, 1], [1, 2, 27, 1], [2, 2, 6, 1], [3, 2, 9, 1], [0, 1, 11, 1], [0, 2, 11, 1], [1, 2, 28, 1], [2, 2, 31, 1], [3, 2, 9, 1], [0, 1, 22, 1], [0, 2, 22, 1], [1, 2, 13, 1], [2, 2, 12, 1], [3, 2, 21, 1], [0, 1, 27, 1], [0, 2, 27, 1], [1, 2, 34, 1], [1, 2, 33, 1], [2, 2, 26, 1], [3, 2, 13, 1], [0, 1, 7, 1], [0, 2, 7, 1], [1, 2, 25, 1], [2, 2, 4, 1], [3, 2, 20, 1], [0, 1, 14, 1], [0, 2, 14, 1], [1, 2, 17, 1], [2, 2, 5, 1], [3, 2, 31, 1], [0, 1, 6, 1], [0, 2, 6, 1], [1, 2, 16, 1], [2, 2, 21, 1], [3, 2, 31, 1], [0, 1, 18, 1], [0, 2, 18, 1], [1, 2, 30, 1], [2, 2, 2, 1], [3, 2, 15, 1], [0, 1, 4, 1], [0, 2, 4, 1], [1, 2, 32, 1], [2, 2, 12, 1], [3, 2, 17, 1], [0, 1, 5, 1], [0, 2, 5, 1], [1, 2, 30, 1], [2, 2, 34, 1], [2, 2, 31, 1], [3, 2, 34, 1], [3, 2, 14, 1], [0, 1, 10, 1], [0, 2, 10, 1], [1, 2, 35, 1], [1, 2, 21, 1], [2, 2, 20, 1], [3, 2, 23, 1], [0, 1, 16, 1], [0, 2, 16, 1], [1, 2, 10, 1], [2, 2, 12, 1], [3, 2, 25, 1]]
    zhuangtaikongzhi = 0
    zhuangtainum = np.zeros((30, 36))
    # print(zhuangtainum)
    for i in range(3,len(recorder)):
        recorder_compute = recorder[:i]
        if recorder_compute[-1][0] == 0:
            if recorder_compute[-1][1] in[1,4,5,40,41,42]:
                # print('recorder_compute',recorder_compute)
                hand, remain, shows, _ = get_data(0, recorder_compute, 0)
                handlist = list(set(hand2card(hand)))
                for card in handlist:
                    s_hu = 0
                    for xunhuan in range(1000):
                        # print('recorder_compute,card',recorder_compute,card)
                        userecorder = copy.deepcopy(recorder_compute)
                        # print('userecorder',userecorder)
                        s = MCTS(userecorder,0,card)
                        if s == 0:
                            s_hu += 1
                    zhuangtainum[zhuangtaikongzhi][card] = s_hu
                print('hand',hand)
                print(zhuangtainum[zhuangtaikongzhi])
                f = open("selfplaylog", "a+")
                f.write('手牌')
                f.write(str(hand) + '\n')
                f.write(str(zhuangtainum[zhuangtaikongzhi]) + '\n')
                f.write('\n')
                zhuangtaikongzhi += 1
    print('消耗时间',time.time()-time_now)

# recorder_to_now = recorder[:2]
    # print(recorder_to_now)
    # for chuci in recorder[2:]:
    #     recorder_to_now.append(chuci)
    #     if chuci[0] == 0:
    #         if chuci[1] == 2 or chuci[1] == 14:
    #             for i in range(1000):
    #                 s = MCTS(recorder_to_now,0,4)
    #                 print(i, s)
    #                 if s == 0:
    #                     num += 1
    #                 if s == 1:
    #                     num1 += 1
    #                 if s == 2:
    #                     num2 += 1
    #                 if s == 3:
    #                     num3 += 1
    #                 print(num, num1, num2, num3)





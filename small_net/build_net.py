# -*- coding: utf-8 -*-
"""
build *.xml files for a network
"""
# TODO
# 使用rl对该信号灯进行控制，多看看原项目代码
# ？增加主干道车道数，可以后延

import numpy as np
import os

MAX_CAR_NUM = 30
SPEED_LIMIT_ST = 20
SPEED_LIMIT_AV = 11
L0 = 200
L0_end = 75

os.chdir(os.path.dirname(__file__))

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def output_nodes(node):
    str_nodes = '<nodes>\n'
    # traffic light nodes
    ind = 1
    for dx in np.arange(0, L0 * 2, L0):
        str_nodes += node % ('nt' + str(ind), dx, 0, 'traffic_light')
        ind += 1
    # other nodes
    ind = 1
    for dx in np.arange(0, L0 * 2, L0):
        str_nodes += node % ('np' + str(ind), dx, -L0_end, 'priority')
        ind += 1
    str_nodes += node % ('np' + str(ind), L0 + L0_end, 0, 'priority')
    ind += 1
    for dx in np.arange(L0, -1, -L0):
        str_nodes += node % ('np' + str(ind), dx, L0_end, 'priority')
        ind += 1
    str_nodes += node % ('np' + str(ind), -L0_end, 0, 'priority')
    ind += 1
    str_nodes += '</nodes>\n'
    return str_nodes

def output_road_types():
    str_types = '<types>\n'
    str_types += '  <type id="a" priority="2" numLanes="2" speed="%.2f"/>\n' % SPEED_LIMIT_ST
    str_types += '  <type id="b" priority="1" numLanes="1" speed="%.2f"/>\n' % SPEED_LIMIT_AV
    str_types += '</types>\n'
    return str_types

def get_edge_str(edge, from_node, to_node, edge_type):
    edge_id = '%s_%s' % (from_node, to_node)
    return edge % (edge_id, from_node, to_node, edge_type)

def output_edges(edge):
    str_edges = '<edges>\n'
    #external roads
    in_edges = [1, 1, 2, 2]
    out_edges = [1, 5, 2, 4]
    for in_i, out_i in zip(in_edges, out_edges):
        in_node = 'nt' + str(in_i)
        out_node = 'np' + str(out_i)
        str_edges += get_edge_str(edge, in_node, out_node, 'b')
        str_edges += get_edge_str(edge, out_node, in_node, 'b')
    in_edges = [1, 2]
    out_edges = [6, 3]
    for in_i, out_i in zip(in_edges, out_edges):
        in_node = 'nt' + str(in_i)
        out_node = 'np' + str(out_i)
        str_edges += get_edge_str(edge, in_node, out_node, 'a')
        str_edges += get_edge_str(edge, out_node, in_node, 'a')

    #internal roads
    from_node = 'nt1'
    to_node = 'nt2'
    str_edges += get_edge_str(edge, from_node, to_node, 'a')
    str_edges += get_edge_str(edge, to_node, from_node, 'a')
    str_edges += '</edges>\n'
    return str_edges

def get_con_str(con, from_node, cur_node, to_node, from_lane, to_lane):
    from_edge = '%s_%s' % (from_node, cur_node)
    to_edge = '%s_%s' % (cur_node, to_node)
    return con % (from_edge, to_edge, from_lane, to_lane)

def get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node):
    str_cons = ''
    # go-through
    str_cons += get_con_str(con, s_node, cur_node, n_node, 0, 0)
    str_cons += get_con_str(con, n_node, cur_node, s_node, 0, 0)
    str_cons += get_con_str(con, w_node, cur_node, e_node, 0, 0)
    str_cons += get_con_str(con, e_node, cur_node, w_node, 0, 0)
    # left-turn
    str_cons += get_con_str(con, s_node, cur_node, w_node, 0, 1)
    str_cons += get_con_str(con, n_node, cur_node, e_node, 0, 1)
    str_cons += get_con_str(con, w_node, cur_node, n_node, 1, 0)
    str_cons += get_con_str(con, e_node, cur_node, s_node, 1, 0)
    # right-turn
    str_cons += get_con_str(con, s_node, cur_node, e_node, 0, 0)
    str_cons += get_con_str(con, n_node, cur_node, w_node, 0, 0)
    str_cons += get_con_str(con, w_node, cur_node, s_node, 0, 0)
    str_cons += get_con_str(con, e_node, cur_node, n_node, 0, 0)
    return str_cons

def output_connections(con):
    str_cons = '<connections>\n'
    # edge nodes
    str_cons += get_con_str_set(con, 'nt1', 'np5', 'np1', 'np6', 'nt2')
    str_cons += get_con_str_set(con, 'nt2', 'np4', 'np2', 'nt1', 'np3')
    
    str_cons += '</connections>\n'
    return str_cons

def output_netconfig():
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <edge-files value="exp.edg.xml"/>\n'
    str_config += '    <node-files value="exp.nod.xml"/>\n'
    str_config += '    <type-files value="exp.typ.xml"/>\n'
    str_config += '    <tllogic-files value="exp.tll.xml"/>\n'
    str_config += '    <connection-files value="exp.con.xml"/>\n'
    str_config += '  </input>\n  <output>\n'
    str_config += '    <output-file value="exp.net.xml"/>\n'
    str_config += '  </output>\n</configuration>\n'
    return str_config

def get_external_od(out_edges, dest=True):
    edge_maps = [0, 1, 2, 2, 2, 1, 1]
    cur_dest = []
    for out_edge in out_edges:
        in_edge = edge_maps[out_edge]
        in_node = 'nt' + str(in_edge)
        out_node = 'np' + str(out_edge)
        if dest:
            edge = '%s_%s' % (in_node, out_node)
        else:
            edge = '%s_%s' % (out_node, in_node)
        cur_dest.append(edge)
    return cur_dest

# def sample_od_pair(orig_edges, dest_edges):
#     from_edges = []
#     to_edges = []
#     for i in range(len(orig_edges)):
#         from_edges.append(np.random.choice(orig_edges[i]))
#         to_edges.append(np.random.choice(dest_edges))
#     return from_edges, to_edges

def init_routes(density):
    init_flow = '  <flow id="i_%s" departPos="random_free" from="%s" to="%s" begin="0" end="1" departLane="%d" departSpeed="0" number="%d" type="type1"/>\n'
    output = ''
    
    # 主干道初始车辆
    car_num_main = int(MAX_CAR_NUM * density)
    # 支路初始车辆(减少为主干道的40%)
    car_num_branch = int(MAX_CAR_NUM * density * 0.4)
    
    # 主干道(东西向): nt1 <-> nt2
    k = 1
    node1 = 'nt1'
    node2 = 'nt2'
    
    # 主干道目的地(优先直行)
    main_sinks = ['nt1_np6', 'nt2_np3']  # 西出口、东出口
    branch_sinks = ['nt1_np5', 'nt1_np1', 'nt2_np2', 'nt2_np4']  # 南北出口
    
    # 主干道车辆,80%直行
    for lane in [0, 1]:
        source_edge = '%s_%s' % (node1, node2)
        # 直行
        sink_edge = np.random.choice(main_sinks, p=[0.5, 0.5])
        output += init_flow % (str(k), source_edge, sink_edge, lane, int(car_num_main * 0.8))
        k += 1
        # 转弯
        sink_edge = np.random.choice(branch_sinks)
        output += init_flow % (str(k), source_edge, sink_edge, lane, int(car_num_main * 0.2))
        k += 1
        
        source_edge = '%s_%s' % (node2, node1)
        # 直行
        sink_edge = np.random.choice(main_sinks, p=[0.5, 0.5])
        output += init_flow % (str(k), source_edge, sink_edge, lane, int(car_num_main * 0.8))
        k += 1
        # 转弯
        sink_edge = np.random.choice(branch_sinks)
        output += init_flow % (str(k), source_edge, sink_edge, lane, int(car_num_main * 0.2))
        k += 1

    return output

def output_flows(peak_flow1, peak_flow2, density, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ext_flow = '  <flow id="f_%s" departPos="random_free" from="%s" to="%s" begin="%d" end="%d" vehsPerHour="%d" type="type1"/>\n'
    str_flows = '<routes>\n'
    str_flows += '  <vType id="type1" length="5" accel="5" decel="10"/>\n'
    
    if density > 0:
        str_flows += init_routes(density)

    # 创建所有可能的源和目的地
    all_srcs = []
    all_srcs.append(get_external_od([4, 5], dest=False))  # 索引0: 南(支路)
    all_srcs.append(get_external_od([6], dest=False))      # 索引1: 西(主干道)
    all_srcs.append(get_external_od([1, 2], dest=False))  # 索引2: 北(支路)
    all_srcs.append(get_external_od([3], dest=False))      # 索引3: 东(主干道)

    all_sinks = []
    all_sinks.append(get_external_od([3]))      # 索引0: 东(主干道)
    all_sinks.append(get_external_od([1, 2]))  # 索引1: 北(支路)
    all_sinks.append(get_external_od([6]))      # 索引2: 西(主干道)
    all_sinks.append(get_external_od([4, 5]))  # 索引3: 南(支路)

    # 定义主干道(东西向)和支路(南北向)
    main_road_src = [1, 3]  # 西、东
    branch_road_src = [0, 2]  # 南、北
    main_road_sink = [0, 2]  # 东、西
    branch_road_sink = [1, 3]  # 北、南

    ratios1 = np.array([0.4, 0.7, 0.9, 1.0, 0.75, 0.5, 0.25])
    ratios2 = np.array([0.3, 0.8, 0.9, 1.0, 0.8, 0.6, 0.2])

    times = np.arange(0, 3001, 300)
    
    for i in range(len(times) - 1):
        t_begin, t_end = times[i], times[i + 1]
        k = 0
        
        for src_idx in range(len(all_srcs)):
            for sink_idx in range(len(all_sinks)):
                # 避免同一侧和掉头
                if src_idx == sink_idx or (src_idx + 2) % 4 == sink_idx:
                    continue
                
                # 基础流量计算
                base_flow = 0
                if i < 7:  # 前 35 分钟
                    if src_idx in [0, 1]:
                        base_flow = peak_flow1 * (0.6 if src_idx == 0 else 1.0) * ratios1[i]
                
                if i >= 3 and i < 10:  # 15-50 分钟
                    if src_idx in [2, 3]:
                        base_flow += peak_flow2 * (0.6 if src_idx == 2 else 1.0) * ratios2[i - 3]
                
                if base_flow <= 0:
                    continue
                
                # 根据道路类型和转向调整流量
                # 判断是否为主干道
                is_main_src = src_idx in main_road_src
                is_main_sink = sink_idx in main_road_sink
                
                # 判断转向类型
                # 直行: 东→西, 西→东 (src_idx=1,sink_idx=2 或 src_idx=3,sink_idx=0)
                is_straight = (src_idx == 1 and sink_idx == 2) or (src_idx == 3 and sink_idx == 0)
                # 左转: 需要跨越对向车道
                is_left = (src_idx == 1 and sink_idx == 1) or (src_idx == 3 and sink_idx == 3) or \
                          (src_idx == 0 and sink_idx == 2) or (src_idx == 2 and sink_idx == 0)
                # 右转: 其他情况
                is_right = not is_straight and not is_left
                
                # 应用流量权重
                flow_val = base_flow
                
                if is_main_src:
                    # 主干道车辆
                    if is_straight:
                        flow_val *= 1.5  # 直行增加50%
                    elif is_left:
                        flow_val *= 0.3  # 左转减少70%
                    elif is_right:
                        flow_val *= 0.4  # 右转减少60%
                else:
                    # 支路车辆 - 整体减少
                    flow_val *= 0.4  # 支路流量减少60%
                
                if flow_val > 0:
                    for e1, e2 in zip(all_srcs[src_idx], all_sinks[sink_idx]):
                        cur_name = str(i) + '_' + str(k)
                        str_flows += ext_flow % (cur_name, e1, e2, t_begin, t_end, int(flow_val))
                        k += 1
    
    str_flows += '</routes>\n'
    return str_flows

def gen_rou_file(path, peak_flow1, peak_flow2, density, seed=None, thread=None):
    if thread is None:
        flow_file = 'exp.rou.xml'
    else:
        flow_file = 'exp_%d.rou.xml' % int(thread)
    write_file(path + flow_file, output_flows(peak_flow1, peak_flow2, density, seed=seed))
    sumocfg_file = path + ('exp_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread))
    return sumocfg_file


def output_config(thread=None):
    if thread is None:
        out_file = 'exp.rou.xml'
    else:
        out_file = 'exp_%d.rou.xml' % int(thread)
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="exp.net.xml"/>\n'
    str_config += '    <route-files value="%s"/>\n' % out_file
    str_config += '    <additional-files value="exp.add.xml"/>\n'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="3600"/>\n'
    str_config += '  </time>\n</configuration>\n'
    return str_config


def get_ild_str(from_node, to_node, ild_str, lane_i=0):
    edge = '%s_%s' % (from_node, to_node)
    return ild_str % (edge, lane_i, edge, lane_i)

def output_ild(ild):
    str_adds = '<additional>\n'
    in_edges = [2, 1, 1, 2, 2, 1]
    out_edges = [3, 6, 1, 2, 4, 5]
    # external edges
    for k, (i, j) in enumerate(zip(in_edges, out_edges)):
        node1 = 'nt' + str(i)
        node2 = 'np' + str(j)
        str_adds += get_ild_str(node2, node1, ild)
        if k < 2:
            str_adds += get_ild_str(node2, node1, ild, lane_i=1)
    # streets
    str_adds += get_ild_str('nt1', 'nt2', ild)
    str_adds += get_ild_str('nt2', 'nt1', ild)
    str_adds += get_ild_str('nt1', 'nt2', ild, lane_i=1)
    str_adds += get_ild_str('nt2', 'nt1', ild, lane_i=1)

    str_adds += '</additional>\n'
    return str_adds

def output_tls(tls, phase):
    str_adds = '<additional>\n'
    # all crosses have 3 phases
    three_phases = ['GGgrrrGGgrrr', 'yyyrrryyyrrr',
                    'rrrGrGrrrGrG', 'rrrGryrrrGry',
                    'rrrGGrrrrGGr', 'rrryyrrrryyr']
    phase_duration = [30, 3]
    for i in range(1, 3):
        node = 'nt' + str(i)
        str_adds += tls % node
        for k, p in enumerate(three_phases):
            str_adds += phase % (phase_duration[k % 2], p)
        str_adds += '  </tlLogic>\n'
    str_adds += '</additional>\n'
    return str_adds

def main():
    # nod.xml file
    node = '  <node id="%s" x="%.2f" y="%.2f" type="%s"/>\n'
    write_file('./exp.nod.xml', output_nodes(node))

    # typ.xml file
    write_file('./exp.typ.xml', output_road_types())

    # edg.xml file
    edge = '  <edge id="%s" from="%s" to="%s" type="%s"/>\n'
    write_file('./exp.edg.xml', output_edges(edge))

    # con.xml file
    con = '  <connection from="%s" to="%s" fromLane="%d" toLane="%d"/>\n'
    write_file('./exp.con.xml', output_connections(con))

    # tls.xml file
    tls = '  <tlLogic id="%s" programID="0" offset="0" type="static">\n'
    phase = '    <phase duration="%d" state="%s"/>\n'
    write_file('./exp.tll.xml', output_tls(tls, phase))

    # net config file
    write_file('./exp.netccfg', output_netconfig())

    # generate net.xml file
    os.system('netconvert -c exp.netccfg')

    # raw.rou.xml file
    write_file('./exp.rou.xml', output_flows(1000, 2000, 0.2))

    # generate rou.xml file
    # os.system('jtrrouter -n exp.net.xml -r exp.raw.rou.xml -o exp.rou.xml')

    # add.xml file
    ild = '  <laneAreaDetector file="ild.out" freq="1" id="%s_%d" lane="%s_%d" pos="-50" endPos="-1"/>\n'
    # ild_in = '  <inductionLoop file="ild_out.out" freq="15" id="ild_in:%s" lane="%s_0" pos="10"/>\n'
    write_file('./exp.add.xml', output_ild(ild))

    # config file
    write_file('./exp.sumocfg', output_config())

if __name__ == '__main__':
    main()
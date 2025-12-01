# TODO
# 东西南北的流量

import numpy as np
import time
# import traci
import libsumo as traci
from collections import deque

class SumoEnvTwoAgents:
    """Single SUMO environment instance"""
    
    def __init__(self,
                sumocfg_path="small_net/exp.sumocfg",
                sumo_bin="sumo",
                port = None,
                label = None,
                delta_time=3.0,
                sim_step_length=1.0,
                warmup_steps=120,
                # peak_threshold=None,
                alpha_low=0.0,
                alpha_high=0.85,
                switch_penalty=1.0,
                use_gui=False,
                max_steps=10000):
        self.sumocfg = sumocfg_path
        self.sumo_bin = "sumo-gui" if use_gui else sumo_bin
        self.port = port
        self.label = label  # libsumo 不支持多实例的 label
        self.delta_time = delta_time
        self.sim_step_length = sim_step_length
        self.warmup_steps = warmup_steps
        # self.peak_threshold = peak_threshold
        self.alpha = [alpha_low, alpha_high]
        self.switch_penalty = switch_penalty
        self.max_steps = max_steps

        # agent <-> tls mapping
        self.agent_tls = {"A": "nt1", "B": "nt2"}
        self.n_actions = 8

        # internal state
        self.traci = None
        self.step_count = 0
        self.prev_phase = {"nt1": None, "nt2": None}
        # self.detectors = []
        
        # 缓存车道信息
        self._lane_cache = {}

        # 东南西北的车流量
        self.ns_flow = []
        self.ew_flow = []

    def _start_sumo(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # libsumo 使用不同的启动方式
                cmd = [
                    "sumo",
                    "-c", self.sumocfg,
                    "--step-length", str(self.sim_step_length),
                    "--no-step-log", "true",
                    "--no-warnings", "true",
                    "--duration-log.disable", "true",
                    "--time-to-teleport", "-1",
                    "--start"
                ]
                
                traci.start(cmd)
                self.traci = traci
                
                # print(f"SUMO started with libsumo (faster!)")
                
                # 预缓存车道信息
                self._cache_lane_info()
                return
                
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5)

    def _cache_lane_info(self):
        """预缓存所有交通灯的车道信息"""
        try:
            for tls in self.agent_tls.values():
                self._lane_cache[tls] = list(self.traci.trafficlight.getControlledLanes(tls))
        except Exception as e:
            print(f"Warning: Failed to cache lane info: {e}")

    def reset(self, full_restart=False):
        if full_restart or not hasattr(self, 'wave_scale'):
            try:
                if self.traci is not None:
                    self.traci.close()
                    time.sleep(0.5)
            except:
                pass


            self._start_sumo()

            self.warm_wave_list = []
            self.warm_wait_list = []

            # try:
            #     self.detectors = list(self.traci.lanearea.getIDList())
            # except:
            #     self.detectors = []
            
            # vals = []
            print(f"Warming up {self.warmup_steps} steps...")
            
            for i in range(self.warmup_steps):
                # print("warmup step:", i+1, "/", self.warmup_steps)
                self.traci.simulationStep()

                wave_tmp = 0
                wait_tmp = 0
                # step_flow = 0

                # for det in self.detectors:
                #     try:
                #         step_flow += self.traci.inductionloop.getLastStepVehicleNumber(det)
                #     except:
                #         pass
                # vals.append(step_flow)

                for tls in self.agent_tls.values():
                    lanes = self._get_controlled_lanes(tls)
                    for l in lanes:
                        try:
                            wave_tmp += self.traci.lane.getLastStepVehicleNumber(l)
                            wait_tmp += self.traci.lane.getWaitingTime(l)
                        except:
                            pass

                self.warm_wave_list.append(wave_tmp)
                self.warm_wait_list.append(wait_tmp)

            # if self.peak_threshold is None:
            #     self.peak_threshold = max(1.0, float(np.percentile(vals, 60)) * 1.5)

            self.wave_scale = max(1.0, np.percentile(self.warm_wave_list, 95))
            self.wait_scale = max(1.0, np.percentile(self.warm_wait_list, 95))
        else:
            try:
                self.traci.load([
                    '-c', self.sumocfg,
                    '--start',
                    '--step-length', str(self.sim_step_length)
                ])
            except Exception as e:
                print(f"Warning: load failed: {e}")
    
        self.step_count = 0
        self.prev_phase = {"nt1": self._safe_get_phase("nt1"), "nt2": self._safe_get_phase("nt2")}
        return self._get_all_obs()

    def _get_controlled_lanes(self, tls):
        """从缓存获取车道"""
        if tls not in self._lane_cache:
            try:
                self._lane_cache[tls] = list(self.traci.trafficlight.getControlledLanes(tls))
            except:
                self._lane_cache[tls] = []
        return self._lane_cache[tls]

    def close(self):
        try:
            if self.traci is not None:
                self.traci.close()
                self.traci = None
        except Exception as e:
            print(f"Error closing libsumo: {e}")

    def _safe_get_phase(self, tls):
        try:
            return int(self.traci.trafficlight.getPhase(tls))
        except:
            return 0

    def _compute_wave(self, tls):
        lanes = self._get_controlled_lanes(tls)
        s = 0
        for l in lanes:
            try:
                s += self.traci.lane.getLastStepVehicleNumber(l)
            except:
                pass
        return float(s)

    def _compute_wait(self, tls):
        lanes = self._get_controlled_lanes(tls)
        s = 0.0
        for l in lanes:
            try:
                s += self.traci.lane.getWaitingTime(l)
            except:
                pass
        return float(s)

    def _compute_downstream_occ(self, tls):
        lanes = self._get_controlled_lanes(tls)
        occs = []
        for l in lanes:
            try:
                links = self.traci.lane.getLinks(l)
                for link in links:
                    try:
                        to_edge = f"{link[0]}"
                        occ = self.traci.lane.getLastStepOccupancy(to_edge) / 100.0
                        occs.append(occ)
                    except:
                        pass
            except:
                pass
        if len(occs) == 0:
            return 0.0
        return float(np.mean(occs))

    # def _compute_total_detector_flow(self):
    #     s = 0
    #     for det in self.detectors:
    #         try:
    #             s += self.traci.lanearea.getLastStepVehicleNumber(det)
    #         except:
    #             pass
    #     return float(s)
    
    def _get_flow(self):
        lanes = self.traci.lane.getIDList()
        print(lanes)
        


    def _get_all_obs(self, neighbor_entropy=None):
        if neighbor_entropy is None:
            neighbor_entropy = {"A": 0.0, "B": 0.0}
        obsA = self._obs_for_agent("A", neighbor_entropy.get("B", 0.0))
        obsB = self._obs_for_agent("B", neighbor_entropy.get("A", 0.0))
        return {"A": obsA, "B": obsB}

    def _obs_for_agent(self, aid, neighbor_entropy=0.0):
        tls = self.agent_tls[aid]
        wave = self._compute_wave(tls)
        wait = self._compute_wait(tls)
        occ = self._compute_downstream_occ(tls)
        # peak_flag = 1.0 if self._compute_total_detector_flow() >= self.peak_threshold else 0.0
        peak_flag = 0.0
        return np.array([wave/self.wave_scale, wait/self.wait_scale, occ, peak_flag, neighbor_entropy], dtype=np.float32)

    def _local_reward(self, aid):
        tls = self.agent_tls[aid]
        q = self._compute_wave(tls)
        w = self._compute_wait(tls)
        occ = self._compute_downstream_occ(tls)
        spill_pen = 10.0 if occ > 0.85 else 0.0
        return -(q + 0.2*w + spill_pen)

    def step(self, action_dict, neighbor_policy_probs=None):
        prev_phases = {tls: self._safe_get_phase(tls) for tls in self.agent_tls.values()}

        for aid, a in action_dict.items():
            tls = self.agent_tls[aid]
            a_int = int(a) % self.n_actions
            try:
                self.traci.trafficlight.setPhase(tls, a_int)
            except:
                pass

        steps = max(1, int(round(self.delta_time / self.sim_step_length)))
        for _ in range(steps):
            self.traci.simulationStep()
            self.step_count += 1

        local = {"A": self._local_reward("A"), "B": self._local_reward("B")}

        for aid in ["A", "B"]:
            tls = self.agent_tls[aid]
            cur_phase = self._safe_get_phase(tls)
            if prev_phases.get(tls, None) is not None and cur_phase != prev_phases[tls]:
                local[aid] -= self.switch_penalty

        # total_flow = self._compute_total_detector_flow()
        # peak_flag = 1.0 if total_flow >= self.peak_threshold else 0.0
        peak_flag = 0.0
        alpha = self.alpha[1] if peak_flag >= 1.0 else self.alpha[0]

        rewards = {}
        rewards["A"] = local["A"] + alpha * local["B"]
        rewards["B"] = local["B"] + alpha * local["A"]

        neighbor_entropy = {}
        if neighbor_policy_probs:
            for aid, probs in neighbor_policy_probs.items():
                p = np.array(probs, dtype=np.float32)
                p = p / (p.sum() + 1e-8)
                neighbor_entropy[aid] = float(-np.sum(p * np.log(p + 1e-8)))
        else:
            neighbor_entropy = {"A": 0.0, "B": 0.0}

        obs = self._get_all_obs(neighbor_entropy)
        done = False
        info = {"alpha": alpha, "peak_flag": peak_flag}

        for tls in self.agent_tls.values():
            self.prev_phase[tls] = self._safe_get_phase(tls)

        done = self.step_count * self.sim_step_length >= self.max_steps

        return obs, rewards, done, info
    
    # def get_phase_info(self, tls_id):
    #     """
    #     获取指定交通灯的当前相位信息
        
    #     Args:
    #         tls_id: 交通灯ID，例如 "nt1" 或 "nt2"
            
    #     Returns:
    #         dict: 包含相位详细信息的字典
    #     """
    #     try:
    #         current_phase_index = self.traci.trafficlight.getPhase(tls_id)
    #         phase_duration = self.traci.trafficlight.getPhaseDuration(tls_id)
    #         phase_state = self.traci.trafficlight.getRedYellowGreenState(tls_id)
    #         program = self.traci.trafficlight.getAllProgramLogics(tls_id)
            
    #         # 获取完整的相位定义
    #         if program:
    #             logic = program[0]  # 通常使用第一个程序
    #             phases = logic.phases
                
    #             phase_info = {
    #                 "tls_id": tls_id,
    #                 "current_phase_index": current_phase_index,
    #                 "current_phase_duration": phase_duration,
    #                 "current_state": phase_state,
    #                 "total_phases": len(phases),
    #                 "all_phases": []
    #             }
                
    #             for idx, phase in enumerate(phases):
    #                 phase_detail = {
    #                     "index": idx,
    #                     "duration": phase.duration,
    #                     "state": phase.state,
    #                     "description": self._describe_phase(phase.state),
    #                     "is_current": (idx == current_phase_index)
    #                 }
    #                 phase_info["all_phases"].append(phase_detail)
                
    #             return phase_info
    #         else:
    #             return {
    #                 "tls_id": tls_id,
    #                 "current_phase_index": current_phase_index,
    #                 "current_state": phase_state,
    #                 "error": "No program logic found"
    #             }
                
    #     except Exception as e:
    #         return {"error": str(e)}
    
    # def _describe_phase(self, state):
    #     """
    #     将相位状态字符串转换为人类可读的描述
        
    #     Args:
    #         state: 相位状态字符串，例如 "rrrrGGGGrrrrrrrrgggggggg"
    #     """
    #     # 24个连接的索引含义
    #     # 0-3: 南北直行, 4-7: 东西直行
    #     # 8-11: 南北左转, 12-15: 东西左转
    #     # 16-19: 南北右转, 20-23: 东西右转
        
    #     descriptions = []
        
    #     # 检查东西直行 (索引 4-7)
    #     ew_straight = state[4:8]
    #     if 'G' in ew_straight:
    #         descriptions.append("东西直行(绿灯)")
    #     elif 'y' in ew_straight or 'Y' in ew_straight:
    #         descriptions.append("东西直行(黄灯)")
        
    #     # 检查南北直行 (索引 0-3)
    #     ns_straight = state[0:4]
    #     if 'G' in ns_straight:
    #         descriptions.append("南北直行(绿灯)")
    #     elif 'y' in ns_straight or 'Y' in ns_straight:
    #         descriptions.append("南北直行(黄灯)")
        
    #     # 检查东西左转 (索引 12-15)
    #     ew_left = state[12:16]
    #     if 'G' in ew_left:
    #         descriptions.append("东西左转(绿灯)")
    #     elif 'y' in ew_left or 'Y' in ew_left:
    #         descriptions.append("东西左转(黄灯)")
        
    #     # 检查南北左转 (索引 8-11)
    #     ns_left = state[8:12]
    #     if 'G' in ns_left:
    #         descriptions.append("南北左转(绿灯)")
    #     elif 'y' in ns_left or 'Y' in ns_left:
    #         descriptions.append("南北左转(黄灯)")
        
    #     # 检查右转 (索引 16-23)
    #     right_turn = state[16:24]
    #     if 'g' in right_turn:
    #         descriptions.append("右转(低优先级绿灯)")
        
    #     return " + ".join(descriptions) if descriptions else "全红"
    
    # def print_all_phases(self, tls_id="nt1"):
    #     """
    #     打印指定交通灯的所有相位信息
        
    #     Args:
    #         tls_id: 交通灯ID，默认为 "nt1"
    #     """
    #     info = self.get_phase_info(tls_id)
        
    #     if "error" in info:
    #         print(f"错误: {info['error']}")
    #         return
        
    #     print(f"\n{'='*80}")
    #     print(f"交通灯: {info['tls_id']}")
    #     print(f"当前相位索引: {info['current_phase_index']}")
    #     print(f"当前相位状态: {info['current_state']}")
    #     print(f"总相位数: {info['total_phases']}")
    #     print(f"{'='*80}\n")
        
    #     for phase in info["all_phases"]:
    #         marker = ">>> " if phase["is_current"] else "    "
    #         print(f"{marker}相位 {phase['index']}: (持续 {phase['duration']}秒)")
    #         print(f"    状态字符串: {phase['state']}")
    #         print(f"    描述: {phase['description']}")
    #         print()

class BatchSumoEnvManager:
    """
    Manager for running multiple SUMO environments in parallel
    Collects time-series data for LSTM training
    """

    def __init__(self, num_envs=4, sumocfg_path="small_net/exp.sumocfg", 
                 sumo_bin="sumo", time_series_len=16, ports=None):
        """
        Args:
            num_envs: Number of parallel SUMO instances
            sumocfg_path: Path to SUMO config
            sumo_bin: SUMO binary path
            base_port: Starting port for SUMO instances
            time_series_len: Length of time-series sequence for LSTM
        """
        self.num_envs = num_envs
        self.time_series_len = time_series_len
        self.envs = []
        self.time_series_buffers = {}  # Per-env time-series buffers
        self.ports = ports
    
        self.labels = [f"sumo_{i}" for i in range(num_envs)]

        for i in range(num_envs):
            label = self.labels[i]
            env = SumoEnvTwoAgents(
                sumocfg_path=sumocfg_path,
                sumo_bin=sumo_bin,
                delta_time=3.0,
                sim_step_length=1.0,
                warmup_steps=120,
                switch_penalty=1.0,
                use_gui=False,
                port=self.ports[i] if self.ports else None,
                label=label
            )
            env._start_sumo()
            print(env._lane_cache)
            self.envs.append(env)
            # Initialize time-series buffer for each agent in each env
            self.time_series_buffers[i] = {
                "A": deque(maxlen=time_series_len),
                "B": deque(maxlen=time_series_len)
            }
        
        print(f"Initialized {num_envs} parallel SUMO environments")

    def reset_all(self, full_restart=False):
        """Reset all environments"""
        obs_list = []
        for i, env in enumerate(self.envs):
            obs = env.reset(full_restart=full_restart or i == 0)
            obs_list.append(obs)
            # Clear time-series buffers
            for aid in ["A", "B"]:
                self.time_series_buffers[i][aid].clear()
                # Initialize buffer with current obs
                for _ in range(self.time_series_len-1):
                    self.time_series_buffers[i][aid].append(np.zeros_like(obs[aid], dtype=np.float32))
                self.time_series_buffers[i][aid].append(obs[aid].copy())
        return obs_list

    def step_all(self, actions_list, neighbor_probs_list=None):
        """
        Step all environments
        
        Args:
            actions_list: List of action dicts, one per environment
            neighbor_probs_list: List of neighbor policy probs, one per environment
            
        Returns:
            obs_list, rewards_list, done_list, info_list, time_series_list
        """
        obs_list = []
        rewards_list = []
        done_list = []
        info_list = []
        time_series_list = []
        
        for i, env in enumerate(self.envs):
            neighbor_probs = neighbor_probs_list[i] if neighbor_probs_list else None
            obs, rewards, done, info = env.step(actions_list[i], neighbor_policy_probs=neighbor_probs)
            
            obs_list.append(obs)
            rewards_list.append(rewards)
            done_list.append(done)
            info_list.append(info)
            
            # Update time-series buffers
            for aid in ["A", "B"]:
                self.time_series_buffers[i][aid].append(obs[aid].copy())
            
            # Extract current time-series as numpy array
            ts_series = {
                "A": np.array(list(self.time_series_buffers[i]["A"]), dtype=np.float32),  # (time_series_len, obs_dim)
                "B": np.array(list(self.time_series_buffers[i]["B"]), dtype=np.float32)
            }
            time_series_list.append(ts_series)
        
        return obs_list, rewards_list, done_list, info_list, time_series_list

    def close_all(self):
        """Close all environments"""
        for env in self.envs:
            env.close()
        print("All SUMO environments closed")

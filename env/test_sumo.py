# 服务器无法display
# 考虑删除w
# 修改协同控制方法，增加上下游交叉口协同放行的时间间隔和时间长度，并通过奖励的对比来决定是否实现协同放行
import numpy as np
import time
# import traci
import libsumo as traci
from collections import deque
import multiprocessing as mp
from multiprocessing import Process, Queue

import sys
sys.path.append("..")
from utils.utils import Normalization #, RewardScaling

class FeatureNormalizer:
    """逐特征独立统计与归一化"""
    def __init__(self, feat_dim=4, shape=(1,)):
        self.feat_dim = feat_dim
        self.norm_list = [Normalization(shape=shape) for _ in range(feat_dim)]

    def update(self, x: np.ndarray):
        for i in range(self.feat_dim):
            _ = self.norm_list[i](np.asarray(x[i], dtype=np.float32), update=True)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float32)
        for i in range(self.feat_dim):
            out[i] = self.norm_list[i](np.asarray(x[i], dtype=np.float32), update=False)
        return out

def env_worker(env_id, cmd_queue, result_queue, sumocfg_path, sumo_bin, time_series_len):
    """
    顶层函数：子进程环境循环，避免 pickle self
    """
    # 在子进程中导入/初始化 SUMO
    # import libsumo as traci  # 如需
    env = SumoEnvTwoAgents(
        sumocfg_path=sumocfg_path,
        sumo_bin=sumo_bin,
        use_gui=False,
        label=f"env_{env_id}"
    )
    try:
        while True:
            cmd, args = cmd_queue.get()
            if cmd == "reset":
                full_restart = args
                obs = env.reset(full_restart=full_restart)
                result_queue.put(("obs", obs))
            elif cmd == "step":
                action_dict, neighbor_probs = args
                obs, rewards, done, info = env.step(
                    action_dict,
                    neighbor_policy_probs=neighbor_probs,
                )
                result_queue.put(("step", (obs, rewards, done, info)))
            elif cmd == "close":
                env.close()
                result_queue.put(("closed", None))
                break
    except Exception as e:
        result_queue.put(("error", str(e)))

class SumoEnvTwoAgents:
    """Single SUMO environment instance"""
    
    def __init__(self,
                sumocfg_path="small_net/exp.sumocfg",
                sumo_bin="sumo",
                label=None,
                sim_step_length=1.0,
                obs_dim=4,
                act_dim=2,
                phase_num=8,
                max_green=60,
                min_green=10,
                switch_penalty=1.0,
                use_gui=False,
                max_steps=10000):
        # sumo config
        self.sumocfg = sumocfg_path
        self.sumo_bin = "sumo-gui" if use_gui else sumo_bin
        self.label = label
        self.sim_step_length = sim_step_length
        self.max_steps = max_steps
        self._lane_cache = {}

        # traffic light parameters and agent config
        self.max_green = max_green
        self.min_green = min_green
        self.agent_tls = {"A": "nt1", "B": "nt2"}
        self.phase_num = phase_num
        self._phase_elapsed = {"nt1": 0.0, "nt2": 0.0}
        
        # obs, act, reward config
        self.switch_penalty = switch_penalty
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # coworking learning parameters


        # obs normalizers
        self.normA = FeatureNormalizer(feat_dim=obs_dim, shape=(1,))
        self.normB = FeatureNormalizer(feat_dim=obs_dim, shape=(1,))
        # reward scaling
        # self.q_scaler = RewardScaling(shape=1, gamma=0.99)
        # self.w_scaler = RewardScaling(shape=1, gamma=0.99)
        # self.q_scaler.reset()
        # self.w_scaler.reset()

        # internal state
        self.traci = None
        self.step_count = 0
        self.prev_phase = {"nt1": None, "nt2": None}
        self.detectors = []

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

        else:
            try:
                self.traci.load([
                    '-c', self.sumocfg,
                    '--start',
                    '--step-length', str(self.sim_step_length),
                    '--no-step-log', 'true',
                    '--no-warnings', 'true',
                    '--duration-log.disable', 'true',
                    '--time-to-teleport', '-1',
                ])
            except Exception as e:
                print(f"Warning: load failed: {e}")
    
        for tls in self.agent_tls.values():
            self.traci.trafficlight.setPhase(tls, 0)
        self.step_count = 0
        self.prev_phase = {"nt1": self._safe_get_phase("nt1"), "nt2": self._safe_get_phase("nt2")}
        # print("initial phases:", self.prev_phase)
        self._phase_elapsed = {"nt1": 0.0, "nt2": 0.0}
        # self.q_scaler.reset()
        # self.w_scaler.reset()
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
        # 获得一个交通灯控制的所有车道的排队车辆数之和
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
    
    def _compute_total_flow(self):
        # 获得网络中所有车道的车辆数之和
        lanes = self.traci.lane.getIDList()
        s = 0
        for l in lanes:
            if ":" not in l:
                s += self.traci.lane.getLastStepVehicleNumber(l)
        return float(s)

    def _get_all_obs(self, neighbor_entropy=None):
        if neighbor_entropy is None:
            neighbor_entropy = {"A": 0.0, "B": 0.0}
        obsA = self._obs_for_agent("A", neighbor_entropy.get("B", 0.0))
        obsB = self._obs_for_agent("B", neighbor_entropy.get("A", 0.0))

        # Update RunningMeanStd with the new observations
        self.normA.update(obsA)
        self.normB.update(obsB)

        # Normalize observations
        obsA_normalized = self.normA.normalize(obsA)
        obsB_normalized = self.normB.normalize(obsB)
        return {"A": obsA_normalized, "B": obsB_normalized}

    def _obs_for_agent(self, aid, neighbor_entropy=0.0):
        tls = self.agent_tls[aid]
        wave = self._compute_wave(tls)
        wait = self._compute_wait(tls)
        occ = self._compute_downstream_occ(tls)
        return np.array([wave, wait, occ, neighbor_entropy], dtype=np.float32)

    # 先用简单的负和作为 reward
    # 之后使用压力作为reward
    def _local_reward(self, aid):
        tls = self.agent_tls[aid]
        q = self._compute_wave(tls)
        w = self._compute_wait(tls)
        qs = q
        ws = w
        occ = self._compute_downstream_occ(tls)
        spill_pen = 2 if occ > 0.85 else 0.0

        # 分别进行 RewardScaling（标准化到 ~O(1)）
        # q_s = float(self.q_scaler(np.asarray(qs, dtype=np.float32)))
        # w_s = float(self.w_scaler(np.asarray(ws, dtype=np.float32)))

        # base = -(q_s + w_s + spill_pen)
        w_s = ws / 10000
        q_s = qs / 100
        reward = -w_s - spill_pen - q_s
        reward_info_dict = {
            "queue": q,
            "waiting_time": w,
            "spill_penalty": spill_pen
        }
        return reward, reward_info_dict

    def _is_transition_phase(self, tls):
        idx = self._safe_get_phase(tls)
        return (idx % 2) == 1
    
    def _advance_phase(self, tls):
        cur_idx = self._safe_get_phase(tls)
        next_idx = (cur_idx + 1) % self.phase_num
        self.traci.trafficlight.setPhase(tls, next_idx)
        self._phase_elapsed[tls] = 0.0

    def step(self, action_dict, neighbor_policy_probs=None):
        prev_phases = {tls: self._safe_get_phase(tls) for tls in self.agent_tls.values()}
        total_flow = self._compute_total_flow()

        # 仿真步进
        self.traci.simulationStep()
        self.step_count += 1
        for tls in self.agent_tls.values():
            self._phase_elapsed[tls] += self.sim_step_length

        # 相位动作控制及转换
        for aid, action in action_dict.items():
            tls = self.agent_tls[aid]
            a_int = int(action) % self.act_dim
            is_trans = self._is_transition_phase(tls)
            elapsed = self._phase_elapsed[tls]
            if not is_trans:
                if elapsed >= self.max_green or (elapsed >= self.min_green and a_int == 1):
                    self._advance_phase(tls)
            else:
                if elapsed >= 3.0:
                    self._advance_phase(tls)

        reward_A, reward_info_A = self._local_reward("A")
        reward_B, reward_info_B = self._local_reward("B")
        local = {"A": reward_A, "B": reward_B}

        for aid in ["A", "B"]:
            tls = self.agent_tls[aid]
            cur_phase = self._safe_get_phase(tls)
            if prev_phases.get(tls, None) is not None and cur_phase != prev_phases[tls]:
                local[aid] -= self.switch_penalty

        rewards = {"A": local["A"], "B": local["B"]}

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
        info = {
            "total_flow": total_flow,
            "reward_info_A": reward_info_A,
            "reward_info_B": reward_info_B
        }

        for tls in self.agent_tls.values():
            self.prev_phase[tls] = self._safe_get_phase(tls)

        done = self.step_count * self.sim_step_length >= self.max_steps

        return obs, rewards, done, info

class BatchSumoEnvManager:
    """
    Manager for running multiple SUMO environments in parallel
    Collects time-series data for LSTM training
    """

    def __init__(self, num_envs=4, sumocfg_path="small_net/exp.sumocfg", 
                 sumo_bin="sumo", time_series_len=16,
                 sim_step_length=1.0, obs_dim=4, act_dim=2):
        """
        Args:
            num_envs: Number of parallel SUMO instances
            sumocfg_path: Path to SUMO config
            sumo_bin: SUMO binary path
            time_series_len: Length of time-series sequence for LSTM
        """
        self.num_envs = num_envs
        self.time_series_len = time_series_len
        self.sim_step_length = sim_step_length
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.time_series_buffers = {}  # Per-env time-series buffers
        self.sumocfg_path = sumocfg_path
        self.sumo_bin = sumo_bin

        self.cmd_queues = [Queue() for _ in range(num_envs)]
        self.result_queues = [Queue() for _ in range(num_envs)]
        self.processes = []
        for i in range(num_envs):
            p = Process(
                target=env_worker,
                args=(i, self.cmd_queues[i], self.result_queues[i],
                      self.sumocfg_path, self.sumo_bin, self.time_series_len),
                daemon=True
            )
            p.start()
            self.processes.append(p)
        
        # 时序缓冲区（主进程维护）
        self.time_series_buffers = {
            i: {"A": deque(maxlen=time_series_len), "B": deque(maxlen=time_series_len)}
            for i in range(num_envs)
        }
        
        print(f"✓ Initialized {num_envs} parallel SUMO environments (multiprocessing)")

    def reset_all(self, full_restart=False):
        """Reset all environments"""
        for q in self.cmd_queues:
            q.put(("reset", full_restart))
        
        # 收集结果
        obs_list = []
        for i in range(self.num_envs):
            msg_type, obs = self.result_queues[i].get()
            if msg_type == "error":
                raise RuntimeError(f"Env {i} error: {obs}")
            obs_list.append(obs)
            
            # 初始化时序缓冲区
            for aid in ["A", "B"]:
                self.time_series_buffers[i][aid].clear()
                for _ in range(self.time_series_len - 1):
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
        # 发送步进命令
        for i in range(self.num_envs):
            neighbor_probs = neighbor_probs_list[i] if neighbor_probs_list else None
            self.cmd_queues[i].put(("step", (actions_list[i], neighbor_probs)))
        
        # 收集结果
        obs_list, rewards_list, done_list, info_list, time_series_list = [], [], [], [], []
        
        for i in range(self.num_envs):
            msg_type, result = self.result_queues[i].get()
            if msg_type == "error":
                raise RuntimeError(f"Env {i} error: {result}")
            
            obs, rewards, done, info = result
            obs_list.append(obs)
            rewards_list.append(rewards)
            done_list.append(done)
            info_list.append(info)
            
            # Update time-series buffers
            for aid in ["A", "B"]:
                self.time_series_buffers[i][aid].append(obs[aid].copy())
            
            # Extract current time-series as numpy array
            ts_series = {
                "A": np.array(list(self.time_series_buffers[i]["A"]), dtype=np.float32),
                "B": np.array(list(self.time_series_buffers[i]["B"]), dtype=np.float32)
            }
            time_series_list.append(ts_series)
        
        return obs_list, rewards_list, done_list, info_list, time_series_list

    def close_all(self):
        """Close all environments"""
        for q in self.cmd_queues:
            q.put(("close", None))
        for i in range(self.num_envs):
            _ = self.result_queues[i].get()
        
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        print("✓ All SUMO environments closed")

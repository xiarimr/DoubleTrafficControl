import numpy as np
import traci
import time
import subprocess

class SumoEnvTwoAgents:
    """
    Agents: "A" -> nt1, "B" -> nt2
    6 phases per TLS
    Includes: switch penalty, peak detection, spatial discount
    """

    def __init__(self,
                sumocfg_path="small_net/exp.sumocfg",
                port=8813,
                sumo_bin="sumo",
                delta_time=4.0,
                sim_step_length=0.1,
                warmup_steps=120,
                peak_threshold=None,
                alpha_low=0.0,
                alpha_high=0.85,
                switch_penalty=1.0,
                use_gui=False):
        self.sumocfg = sumocfg_path
        self.port = port
        self.sumo_bin = "sumo-gui" if use_gui else sumo_bin
        self.delta_time = delta_time
        self.sim_step_length = sim_step_length
        self.warmup_steps = warmup_steps
        self.peak_threshold = peak_threshold
        self.alpha = [alpha_low, alpha_high]
        self.switch_penalty = switch_penalty

        # agent <-> tls mapping (fixed)
        self.agent_names = ["A", "B"]
        self.agent_tls = {"A": "nt1", "B": "nt2"}
        self.n_actions = 6

        # internal state
        self.traci = None
        self.step_count = 0
        self.prev_phase = { "nt1": None, "nt2": None }  # for switch detection
        self.detectors = []

        # start sumo
        self._start_sumo()
        # warmup & initialize detectors etc.
        self.reset()

    def _start_sumo(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                port = self.port + attempt
                cmd = [self.sumo_bin, "-c", self.sumocfg, "--start", 
                    "--step-length", str(self.sim_step_length), 
                    "--remote-port", str(port)]
                self.sumo_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                                stderr=subprocess.PIPE)
                time.sleep(1.0)
                traci.init(port)
                self.traci = traci
                self.port = port
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.0)

    def reset(self, full_restart=False):
        """
        Reset the simulation environment.
        
        Args:
            full_restart: If True, completely restart SUMO process and redo warmup.
                         If False, use fast reset via traci.load() (default for training).
        """
        # Only do full restart if requested or if not yet initialized
        if full_restart or not hasattr(self, 'wave_scale'):
            # Full restart: close connection and restart SUMO process
            try:
                if self.traci is not None:
                    self.traci.close()
            except:
                pass
            
            # Kill old SUMO process to prevent resource leaks
            try:
                if hasattr(self, 'sumo_proc') and self.sumo_proc is not None:
                    self.sumo_proc.terminate()
                    self.sumo_proc.wait(timeout=3.0)
            except:
                try:
                    if hasattr(self, 'sumo_proc') and self.sumo_proc is not None:
                        self.sumo_proc.kill()
                except:
                    pass
            
            # Start new SUMO session
            cmd = [self.sumo_bin, "-c", self.sumocfg, "--start", "--step-length", str(self.sim_step_length), "--remote-port", str(self.port)]
            self.sumo_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2.0)
            traci.init(self.port)
            self.traci = traci
            
            # Initialize warmup data lists
            self.warm_wave_list = []
            self.warm_wait_list = []
            
            # Get detectors
            try:
                self.detectors = list(self.traci.inductionloop.getIDList())
            except:
                self.detectors = []
            vals = []  # collect flows every warmup step

            # Warmup phase
            for _ in range(self.warmup_steps):
                self.traci.simulationStep()

                wave_tmp = 0
                wait_tmp = 0
                step_flow = 0

                for det in self.detectors:
                    try:
                        step_flow += self.traci.inductionloop.getLastStepVehicleNumber(det)
                    except:
                        pass
                vals.append(step_flow)

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

            # Compute normalization scales (only during full restart)
            if self.peak_threshold is None:
                self.peak_threshold = max(
                    1.0,
                    float(np.percentile(vals, 60)) * 1.5
                )

            self.wave_scale = max(1.0, np.percentile(self.warm_wave_list, 95))    
            self.wait_scale = max(1.0, np.percentile(self.warm_wait_list, 95))
        else:
            # Fast reset: just reload the simulation without restarting process
            try:
                self.traci.load(['-c', self.sumocfg, '--start', '--step-length', str(self.sim_step_length)])
            except Exception as e:
                # If load fails, log and continue from current state
                print(f"Warning: traci.load() failed: {e}. Continuing with current simulation state.")
                pass
        
        # Reset step counter and phase tracking
        self.step_count = 0
        self.prev_phase = { "nt1": self._safe_get_phase("nt1"), "nt2": self._safe_get_phase("nt2") }

        return self._get_all_obs()

    def close(self):
        """Close SUMO connection and terminate process properly."""
        # Close traci connection first
        try:
            if self.traci is not None:
                self.traci.close()
        except:
            pass
        
        # Terminate SUMO process gracefully
        try:
            if hasattr(self, 'sumo_proc') and self.sumo_proc is not None:
                self.sumo_proc.terminate()
                self.sumo_proc.wait(timeout=3.0)
        except:
            # Force kill if terminate doesn't work
            try:
                if hasattr(self, 'sumo_proc') and self.sumo_proc is not None:
                    self.sumo_proc.kill()
            except:
                pass

    # ---------- helpers ----------
    def _safe_get_phase(self, tls):
        try:
            return int(self.traci.trafficlight.getPhase(tls))
        except:
            return 0

    def _get_controlled_lanes(self, tls):
        try:
            return list(self.traci.trafficlight.getControlledLanes(tls))
        except:
            return []

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
                # links is list of tuples (toEdge, toLaneIndex, ...) — we try safe reading
                for link in links:
                    try:
                        # link could be tuple, take first as edge id then lane index 0
                        to_edge = f"{link[0]}_{link[1]}"
                        occ = self.traci.lane.getLastStepOccupancy(to_edge) / 100.0
                        occs.append(occ)
                    except:
                        pass
            except:
                pass
        if len(occs) == 0:
            return 0.0
        return float(np.mean(occs))

    def _compute_total_detector_flow(self):
        s = 0
        for det in self.detectors:
            try:
                s += self.traci.inductionloop.getLastStepVehicleNumber(det)
            except:
                pass
        return float(s)

    # ---------- obs / reward / step ----------
    def _get_all_obs(self, neighbor_entropy=None):
        if neighbor_entropy is None:
            neighbor_entropy = {"A":0.0, "B":0.0}
        # Each agent observes its NEIGHBOR's entropy (cross-observation)
        obsA = self._obs_for_agent("A", neighbor_entropy.get("B", 0.0))
        obsB = self._obs_for_agent("B", neighbor_entropy.get("A", 0.0))
        return {"A": obsA, "B": obsB}

    def _obs_for_agent(self, aid, neighbor_entropy=0.0):
        tls = self.agent_tls[aid]
        wave = self._compute_wave(tls)
        wait = self._compute_wait(tls)
        occ = self._compute_downstream_occ(tls)
        peak_flag = 1.0 if self._compute_total_detector_flow() >= self.peak_threshold else 0.0
        # normalized obs vector
        return np.array([wave/self.wave_scale, wait/self.wait_scale, occ, peak_flag, neighbor_entropy], dtype=np.float32)

    def _local_reward(self, aid):
        tls = self.agent_tls[aid]
        q = self._compute_wave(tls)
        w = self._compute_wait(tls)
        occ = self._compute_downstream_occ(tls)
        spill_pen = 10.0 if occ > 0.85 else 0.0
        # switch penalty (if action changed it will be applied outside when we check prev_phase)
        return -(q + 0.2*w + spill_pen)

    def step(self, action_dict, neighbor_policy_probs=None):
        """
        action_dict: {"A": int, "B": int}
        neighbor_policy_probs: optional dict of probs for entropy fingerprint
        """
        # get previous phases for switch detection
        prev_phases = {tls: self._safe_get_phase(tls) for tls in self.agent_tls.values()}

        # apply actions
        for aid, a in action_dict.items():
            tls = self.agent_tls[aid]
            a_int = int(a) % self.n_actions
            try:
                self.traci.trafficlight.setPhase(tls, a_int)
            except:
                pass

        # simulate delta_time seconds (use step length)
        steps = max(1, int(round(self.delta_time / self.sim_step_length)))
        for _ in range(steps):
            self.traci.simulationStep()
            self.step_count += 1

        # compute local rewards
        local = {"A": self._local_reward("A"), "B": self._local_reward("B")}

        # apply switch penalty: check current phases vs prev_phases
        for aid in ["A","B"]:
            tls = self.agent_tls[aid]
            cur_phase = self._safe_get_phase(tls)
            if prev_phases.get(tls, None) is not None and cur_phase != prev_phases[tls]:
                # penalize switch
                local[aid] -= self.switch_penalty

        # compute alpha via peak detection
        total_flow = self._compute_total_detector_flow()
        peak_flag = 1.0 if total_flow >= self.peak_threshold else 0.0
        alpha = self.alpha[1] if peak_flag >= 1.0 else self.alpha[0]

        # spatially discounted reward (simple two-node distance)
        rewards = {}
        rewards["A"] = local["A"] + alpha * local["B"]
        rewards["B"] = local["B"] + alpha * local["A"]

        # neighbor entropy for obs fingerprint
        neighbor_entropy = {}
        if neighbor_policy_probs:
            for aid, probs in neighbor_policy_probs.items():
                p = np.array(probs, dtype=np.float32)
                p = p / (p.sum() + 1e-8)
                neighbor_entropy[aid] = float(-np.sum(p * np.log(p + 1e-8)))
        else:
            neighbor_entropy = {"A":0.0, "B":0.0}

        obs = self._get_all_obs(neighbor_entropy)
        done = False
        info = {"alpha": alpha, "peak_flag": peak_flag}

        # update prev_phase
        for tls in self.agent_tls.values():
            self.prev_phase[tls] = self._safe_get_phase(tls)

        return obs, rewards, done, info

import traci
import time
import sys
import os
import socket
from contextlib import closing

# --- 辅助函数：查找可用端口 ---
def find_free_port(start_port=8813, num_ports=4):
    """
    查找指定数量的可用端口
    """
    free_ports = []
    current_port = start_port
    
    while len(free_ports) < num_ports:
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.bind(('127.0.0.1', current_port))
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                free_ports.append(current_port)
        except OSError:
            pass  # 端口被占用，跳过
        current_port += 1
    
    return free_ports

# --- 1. 配置 ---
num_sumos = 4
base_port = 8813
sumo_labels = []
ports = []
log_dir = "sumo_logs"
os.makedirs(log_dir, exist_ok=True)

# 【修正】自动查找可用端口
print("--- Finding available ports ---")
ports = find_free_port(start_port=base_port, num_ports=num_sumos)
print(f"Available ports: {ports}\n")

# --- 2. 启动所有SUMO实例 ---
print("--- Starting all SUMO instances with logging ---")
for i in range(num_sumos):
    port = ports[i]
    label = f"sumo_{i}"
    log_file = os.path.join(log_dir, f"{label}.log")

    # 不添加 --remote-port 参数（避免重复）
    cmd = ["sumo", 
           "-c", "small_net/exp.sumocfg",
           "--log-file", log_file,
           "--start"]

    try:
        # 【修正】移除 wait 参数，只使用 port 和 label
        traci.start(cmd, port=port, label=label)
        sumo_labels.append(label)
        print(f"✓ Successfully started '{label}' on port {port}")
        time.sleep(1)  # 给SUMO进程稳定的时间
    except Exception as e:
        print(f"✗ Error starting '{label}' on port {port}: {e}")
        print(f"  Please check log file: {log_file}")
        # 退出前关闭所有已成功启动的实例
        for l in sumo_labels:
            try:
                traci.close(label=l)
            except:
                pass
        sys.exit("Aborting due to failed SUMO start.")

print(f"\n✓ All {num_sumos} SUMO instances are running on ports {ports}\n")

# --- 3. 并行执行仿真步骤 ---
print("--- Starting simulation steps ---\n")
try:
    for step in range(100):
        for idx, label in enumerate(sumo_labels):
            traci.switch(label)
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            
            # 只在第一步打印时间
            if step == 0:
                print(f"[{label} - Port {ports[idx]}] Step {step+1}: Simulation time is {current_time}")

        if (step + 1) % 10 == 0:
            print(f"--- Simulation step {step+1} completed for all instances ---")
        
        time.sleep(0.01)
        
except Exception as e:
    print(f"\n✗ Error during simulation: {e}")
    print("Check the log files in the 'sumo_logs' directory for details.")
finally:
    # --- 4. 关闭所有连接 ---
    print("\n--- Closing all connections ---")
    for label in sumo_labels:
        try:
            traci.close(label=label)
        except:
            pass
    print("✓ All SUMO instances closed.")
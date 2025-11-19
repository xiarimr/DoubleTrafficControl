import traci
import time
import sys
import os

# --- 1. 配置 ---
num_sumos = 4
base_port = 8813
sumo_labels = []
log_dir = "sumo_logs"
os.makedirs(log_dir, exist_ok=True) # 创建日志文件夹

# --- 2. 启动所有SUMO实例 ---
print("--- Starting all SUMO instances with logging ---")
for i in range(num_sumos):
    port = base_port + i
    label = f"sumo_{i}"
    log_file = os.path.join(log_dir, f"{label}.log") # 为每个实例指定日志文件

    # 【诊断修改】添加 --log-file 参数来捕获SUMO的内部输出
    cmd = ["sumo", 
           "-c", "small_net/exp.sumocfg",
           "--log-file", log_file,
           "--start"]

    try:
        # traci.start 会自动处理 --remote-port
        traci.start(cmd, port=0, label=label)
        sumo_labels.append(label)
        print(f"Successfully started and connected to '{label}' on port {port}.")
    except Exception as e:
        print(f"!!! Error starting '{label}' on port {port}: {e}")
        print(f"--- Please check the log file for details: {log_file} ---")
        # 退出前关闭所有已成功启动的实例
        for l in sumo_labels:
            try: traci.close(label=l)
            except: pass
        sys.exit("Aborting due to failed SUMO start.")

print("\n--- All SUMO instances are running. Starting simulation steps. ---\n")

# --- 3. 并行执行仿真步骤 ---
try:
    for step in range(100):
        for label in sumo_labels:
            traci.switch(label)
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            # 为了减少输出，只在第一步打印时间
            if step == 0:
                print(f"[{label}] Step {step+1}: Simulation time is {current_time}")

        if (step + 1) % 10 == 0:
            print(f"--- Simulation step {step+1} completed for all instances. ---")
        time.sleep(0.01)
except Exception as e:
    print(f"\n!!! An error occurred during simulation: {e}")
    print("This might be a 'peer shutdown' error. Check the log files in the 'sumo_logs' directory.")
finally:
    # --- 4. 关闭所有连接 ---
    print("\n--- Simulation finished or failed. Closing all connections. ---")
    for label in sumo_labels:
        try:
            traci.switch(label)
            traci.close()
        except:
            pass # 如果连接已关闭，则忽略
    print("All SUMO instances closed.")

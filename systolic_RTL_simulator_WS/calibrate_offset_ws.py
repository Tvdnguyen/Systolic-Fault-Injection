
import os
import subprocess
import numpy as np
import shutil

def run_calibration():
    N = 8
    # 1. Generate Input/Weight files
    # We want to check PE[0][0].
    # Weight Matrix: 1 at [0,0], 0 elsewhere.
    # Input Matrix: 1 at [0,0] (time 0, row 0), 0 elsewhere.
    
    # Weight File
    # TB Logic:
    # for (r = N-1; r >= 0; r = r - 1) ... m1[c] = mem_weights[r*N + c];
    # So flattened index [r*N + c]. We want r=0, c=0.
    with open("weight.mem", "w") as f:
        for r in range(N):
            for c in range(N):
                val = 1 if (r == 0 and c == 0) else 0
                f.write(f"{val:02x}\n")

    # Act File
    # TB Logic:
    # m0[r] = mem_acts[t*N + r];
    # We want t=0, r=0 to be 1.
    with open("act.mem", "w") as f:
        # Generate enough cycles. Say 50.
        for t in range(50):
            for r in range(N):
                val = 1 if (t == 0 and r == 0) else 0
                f.write(f"{val:02x}\n")
                
    # 2. Compile
    cmd = [
        "iverilog", "-g2012",
        "-P", "dut_tb_ws.N=8",
        "-o", "sim_calib",
        "dut_tb_ws.sv", 
        "systolic_ws.sv",
        "systolic_ws_faulty.sv",
        "pe_ws.v", 
        "pe_ws_faulty.v",
        "pe_control_ws.v",
        "pe_datapath_ws.v",
        "pe_datapath_ws_faulty.v" 
    ]
    print("Compiling...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    # 3. Run
    # Phase 1 (Load) takes N cycles (8).
    # Reset is 20ns? TB Cycle logic: always #5 clk. Period 10.
    # TB counts cycles based on posedge.
    # Load Weights: Phase 1 loop N times.
    # Then some wait.
    # Then Phase 2 starts.
    # cycle_count logic: increments on posedge.
    
    print("Running Simulation...")
    cmd_run = ["vvp", "sim_calib", "+MAX_CYCLES=50", "+OUTPUT_FILE=output_calib.txt"]
    subprocess.run(cmd_run, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 4. Analyze Output
    print("Analyzing Output...")
    found = False
    with open("output_calib.txt", "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2: continue
            
            try:
                cycle = int(parts[0])
                # Gold outputs are parts[1] to parts[N] (1..8) for Col 0..7
                # We check Col 0 (parts[1])
                res_col0 = int(parts[1])
                
                if res_col0 != 0:
                    print(f"Found VALID Output at Cycle: {cycle} (Value: {res_col0})")
                    print(f"This corresponds to PE[0][0] output.")
                    
                    # Formula: Output_Cycle = OFFSET + Row + Col
                    # Here Row=0, Col=0.
                    # OFFSET = Output_Cycle
                    
                    print(f"Calculated OFFSET = {cycle}")
                    found = True
                    break
            except:
                pass
                
    if not found:
        print("No output found! Simulation might have failed or latency is > 50.")

if __name__ == "__main__":
    run_calibration()

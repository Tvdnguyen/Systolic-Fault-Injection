
import os
import subprocess
import glob

# Config
SIM_DIR = "."
BINARY = "sim_campaign_ws"
COMPILE_CMD_TEMPLATE = "iverilog -g2012 -P dut_tb_ws.FAULT_ROW={r} -P dut_tb_ws.FAULT_COL={c} -P dut_tb_ws.FAULT_TARGET=1 -o {bin} dut_tb_ws.sv systolic_ws.sv systolic_ws_faulty.sv pe_ws.v pe_ws_faulty.v pe_control_ws.v pe_datapath_ws.v pe_datapath_ws_faulty.v"
# TARGET=1 is WEIGHT (Stationary) in fixed ver10.

def run_test():
    print("--- Verifying WS Cycle 19 on All PEs ---")
    
    # 1. Generate Input Data Once (Reuse weight/act for consistency?)
    # Actually, we need to generate valid weight/act for Conv1.
    # Let's rely on run_fault_campaign_ver10.py to generate them?
    # Or just use random data here, since we look for DIFF.
    # Random is fine.
    
    # Generate weight.mem (8x8 transposed -> 64)
    with open("weight.mem", "w") as f:
        for _ in range(64): f.write("01\n") # All 1s
    
    # Generate act.mem (Scanner)
    # 200 lines sufficient for Cycle 19
    with open("act.mem", "w") as f: # act.mem is stream inputs
        for i in range(200*8): f.write("02\n") # All 2s
        
    results = []
    
    for r in range(8):
        for c in range(8):
            # Compile for this PE
            bin_name = f"sim_pe_{r}_{c}"
            compile_cmd = COMPILE_CMD_TEMPLATE.format(r=r, c=c, bin=bin_name).split()
            subprocess.run(compile_cmd, check=True, stdout=subprocess.DEVNULL)
            
            # Run Simulation at Cycle 19
            # Global Cycle 19.
            cmd = [
                "vvp", bin_name,
                "+FAULT_ENABLE=1",
                f"+FAULT_ROW={r}",
                f"+FAULT_COL={c}",
                "+FAULT_CYCLE=19",
                "+FAULT_VAL=255", # Flip bits to ensure diff
                "+MAX_CYCLES=50",
                "+OUTPUT_FILE=output.txt",
                "+MEM_WEIGHT=weight.mem",
                "+MEM_ACT=act.mem"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            
            if not os.path.exists("output.txt"):
                print(f"PE({r},{c}): No Output")
                continue
                
            formatted_diffs = []
            with open("output.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.replace(',', ' ').split()
                    if len(parts) > 1:
                        cycle = int(parts[0]) # flat_idx
                        vals = [int(x) for x in parts[1:]]
                        mid = len(vals)//2
                        gold = vals[:mid]
                        fault = vals[mid:]
                        
                        if gold != fault:
                            # Find faulty columns (local_col / ch_local)
                            for ch_local in range(8):
                                if gold[ch_local] != fault[ch_local]:
                                    # APPLY WS MAPPING LOGIC (Conv1, Tile 0)
                                    OFFSET = 18
                                    local_row = cycle - ch_local - OFFSET
                                    
                                    if local_row >= 0:
                                        # Tile 0: g_ch = ch_local, g_px = local_row
                                        g_ch = ch_local
                                        g_px = local_row
                                        
                                        # Map to image
                                        g_row = g_px // 32
                                        g_col_img = g_px % 32
                                        
                                        formatted_diffs.append(f"({g_ch}, {g_row}, {g_col_img})")
                                        
            if formatted_diffs:
                # Deduplicate
                formatted_diffs = sorted(list(set(formatted_diffs)))
                results.append(f"PE({r},{c}): {formatted_diffs}")
            else:
                # If diff exists but local_row < 0 (Invalid), note it
                pass
                
            # Clean binary
            if os.path.exists(bin_name): os.remove(bin_name)

    print("\n--- RESULTS ---")
    for res in results:
        print(res)
        
    print("\n--- THEORY CHECK ---")
    print("In WS, PE(r,c) holds Weight(r,c).")
    print("Fault should affect Output Column 'c'.")
    print("Does the result show 'Cols [c]'?")

if __name__ == "__main__":
    run_test()

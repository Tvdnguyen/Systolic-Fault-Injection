
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import subprocess
import os
import sys
import argparse
import shutil
import random
import multiprocessing
import time
import json

# Import Model (Ensure traffic_sign_net_small.py is in PYTHONPATH or current dir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Adjust path if needed
# Try standardized import
try:
    from traffic_sign_net_small import TrafficSignNetSmallNoDropout
except ImportError:
    # Fallback/Mock if running standalone without repo structure
    class TrafficSignNetSmallNoDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1)

# ==============================================================================
# CONFIG
# ==============================================================================
DATAFLOW = "OS"
N = 8    
M_SIM = 24 # Padded Dimension for Banked Memory (Verification Approved)
NUM_PROCESSES = min(32, os.cpu_count() or 8)

TARGET_MAP = {
    "INPUT": 0,   # Physical Vertical (m1/in_b) -> Vertical Propagation
    "WEIGHT": 1,  # Physical Horizontal (m0/in_a) -> Horizontal Propagation
    "PSUM": 2
}

def ensure_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except [FileExistsError, OSError]:
            pass

# ==============================================================================
# DATA GENERATION (BANKED MEMORY)
# ==============================================================================
def generate_mem_files(mat_a, mat_b, prefix=""):
    """
    Generate m0.mem (Inputs) and m1.mem (Weights) in Banked Format for OS.
    mat_a: [N, N] (Inputs)
    mat_b: [N, N] (Weights - Actually Test Logic treats B as Weights usually)
    """
    # Pad to M_SIM x M_SIM
    A_pad = np.zeros((M_SIM, M_SIM), dtype=int)
    B_pad = np.zeros((M_SIM, M_SIM), dtype=int)
    
    # Fill N x N region
    # Torch -> Numpy
    if isinstance(mat_a, torch.Tensor): mat_a = mat_a.numpy()
    if isinstance(mat_b, torch.Tensor): mat_b = mat_b.numpy()
    
    A_pad[:N, :N] = mat_a
    B_pad[:N, :N] = mat_b
    
    # Filenames
    m0_name = f"{prefix}m0.mem"
    m1_name = f"{prefix}m1.mem"
    
    # 1. Write m0.mem (Matrix A - Inputs) -> Banked Interleaved
    # Bank 0: Row 0, 8, 16...
    banks_m0 = [[] for _ in range(N)]
    for r in range(M_SIM):
        bank_idx = r % N
        for c in range(M_SIM):
            banks_m0[bank_idx].append(A_pad[r, c])
            
    with open(m0_name, "w") as f:
        for b in range(N):
            for val in banks_m0[b]:
                f.write(f"{int(val):02x}\n")
 
    # 2. Write m1.mem (Matrix B - Weights) -> Transposed -> Banked Interleaved
    # OS Testbench Logic: Weights flow vertically (loaded into vertical banks?)
    # But TB expects m1.mem to be loaded into m1_mem which is N banks.
    # Logic from verify_os_fault_pattern.py: TRANSPOSING is required.
    B_T = B_pad.T
    banks_m1 = [[] for _ in range(N)]
    for r in range(M_SIM): # r of B_T is Column of B
        bank_idx = r % N
        for c in range(M_SIM):
            banks_m1[bank_idx].append(B_T[r, c])
            
    with open(m1_name, "w") as f:
        for b in range(N):
            for val in banks_m1[b]:
                f.write(f"{int(val):02x}\n")
                
    return m0_name, m1_name

# ==============================================================================
# SIMULATION WORKER
# ==============================================================================
def compile_simulation(f_row, f_col, f_target):
    """
    Compiles a unique binary for the specific PE and Target.
    Returns the filename of the compiled binary.
    """
    binary_name = f"sim_campaign_os_{f_row}_{f_col}_{f_target}"
    
    cmd = [
        "iverilog", "-g2012",
        f"-Pdut_tb_dual_os.N={N}",
        f"-Pdut_tb_dual_os.M={M_SIM}",
        f"-Pdut_tb_dual_os.FAULT_ROW={f_row}",
        f"-Pdut_tb_dual_os.FAULT_COL={f_col}",
        f"-Pdut_tb_dual_os.FAULT_TARGET={f_target}",
        "-o", binary_name,
        "dut_tb_dual_os.sv", 
        "systolic_os_faulty.sv", "systolic.sv",
        "pe_os_faulty.v", "pe_datapath_os_faulty.v",
        "pe_v2.v", "pe_datapath_v2_comb_1.v", "pe_control_v2.v",
        "add_shift_multiplier_simple.v", "mem_read_m0.sv", "mem_read_m1.sv",
        "counter.v", "pipe.sv"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return binary_name

def worker_simulation(args):
    """
    Worker function.
    """
    g_cyc, r, c, tile_duration, mat_a, mat_b, pe_dir_name, t_dir_name, save_dir, out_ch, num_tiles_n, binary_name = args
    
    job_id = f"{os.getpid()}_{g_cyc}"
    output_file = f"output_{job_id}.txt"
    
    tmp_dir = os.path.join("tmp_workers_os", job_id)
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        # Copy the specific binary for this PE
        # Assuming binary is in the parent/current directory
        shutil.copy(os.path.join(os.getcwd(), binary_name), os.path.join(tmp_dir, "sim_campaign_os"))
        cwd_backup = os.getcwd()
        os.chdir(tmp_dir)
        
        # 1. Generate Data
        generate_mem_files(mat_a, mat_b, prefix="")
        
        # 2. Run
        max_cyc = tile_duration + 50
        local_cyc = g_cyc % tile_duration
        
        # Adjust local_cyc for OS skew?
        # In WS, we injected at Exact Cycle. 
        # In OS, logical cycle vs physical cycle depends.
        # But we simply inject at the requested cycle relative to tile start.
        
        cmd_run = [
            "vvp", "sim_campaign_os",
            f"+FAULT_ENABLE=1",
            f"+FAULT_ROW={r}",
            f"+FAULT_COL={c}",
            f"+FAULT_CYCLE={local_cyc}",
            f"+FAULT_VAL=100", # Significant value to be safe
            f"+MAX_CYCLES={max_cyc}",
            f"+OUTPUT_FILE={output_file}"
        ]
        
        subprocess.run(cmd_run, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. Parse Output
        valid_lines = []
        if os.path.exists(output_file):
            valid_vector_idx = 0 # Track sequential valid outputs for local_col mapping
            with open(output_file, "r") as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 2 + 2*N: continue
                    if 'x' in line: continue
                    
                    try:
                        # Format: Cycle, ValidG, ValidF, G0..7, F0..7
                        val_g = int(parts[1], 16)
                        if val_g == 0: continue
                        
                        # Handle OS Bubble Pattern (Data, Bubble, Data...) due to 1-cycle latency gap
                        # Output stream is 16 cycles for 8 logical values.
                        # Map 0,1 -> Col N-1; 2,3 -> Col N-2, ... (Reverse Draining)
                        local_col = (N - 1) - ((valid_vector_idx // 2) % N)
                        valid_vector_idx += 1
                        
                        g_vec = np.array([int(x) for x in parts[3 : 3+N]])
                        f_vec = np.array([int(x) for x in parts[3+N : 3+2*N]])
                        
                        diff = np.abs(g_vec - f_vec)
                        
                        # Check for diffs
                        for idx, d_val in enumerate(diff):
                            if d_val > 0:
                                # idx is Local Row (output vector element)
                                local_row = idx
                                
                                # Global Coord Calc:
                                t_idx = g_cyc // tile_duration
                                tile_m_idx = t_idx // num_tiles_n # Channel Block
                                tile_n_idx = t_idx % num_tiles_n  # Pixel Block
                                
                                global_ch = (tile_m_idx * N) + local_row
                                
                                # Global Pixel Index with Column Offset
                                global_px = (tile_n_idx * N) + local_col
                                
                                # Map Pixel Index to Image Row/Col (32x32)
                                img_row = global_px // 32
                                img_col = global_px % 32
                                
                                if global_ch < out_ch:
                                    valid_lines.append(f"{global_ch}, {img_row}, {img_col}\n")
                                    
                    except ValueError:
                        continue

        # 4. Save Results
        if len(valid_lines) > 0:
             fname = f"faulty_{pe_dir_name}_{t_dir_name}_cycle_{g_cyc}.txt"
             abs_save_dir = os.path.join(cwd_backup, save_dir)
             fpath = os.path.join(abs_save_dir, fname)
             
             # Unique entries to save space
             unique_lines = list(set(valid_lines))
             with open(fpath, "w") as f:
                 for l in unique_lines: f.write(l)
             return (g_cyc, 1) # 1 file created
             
    except Exception:
        pass
    finally:
        os.chdir(cwd_backup)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
    return (g_cyc, 0)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=str, default="conv1")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--exhaustive", action="store_true", help="Run all active cycles sequentially")
    args = parser.parse_args()
    
    layer_name = args.layer
    target_samples = args.samples
    
    print(f"=== VER10: OS CAMPAIGN ({layer_name}) ===")
    
    # 1. Model Info
    LAYER_DIMS = {'conv1': (32, 32), 'conv2': (32, 32), 'conv3': (16, 16)}
    H_out, W_out = LAYER_DIMS.get(layer_name, (32, 32))
    
    model = TrafficSignNetSmallNoDropout()
    try:
        layer_obj = dict(model.named_modules())[layer_name]
    except KeyError:
        # Fallback for conv1
        layer_obj = model.conv1
        
    out_ch = layer_obj.out_channels
    N_pixels = H_out * W_out
    
    # Tiling info
    num_tiles_m = (out_ch + N - 1) // N
    num_tiles_n = (N_pixels + N - 1) // N
    total_tiles = num_tiles_m * num_tiles_n
    
    # Timing
    # OS Timing: Load (Stream) + Compute (L) + Drain (2N)
    # Must match Profiler's Assumption for "Global Cycle" alignment
    L_SCAN = 100 
    TILE_DURATION = 130 
    if layer_name in ['conv3', 'conv5']:
         TILE_DURATION = 200 
    
    # 2. Generate Tile Data
    TILES_DATA = []
    for _ in range(total_tiles):
        A = torch.randint(1, 64, (N, N))
        B = torch.randint(1, 64, (N, N))
        TILES_DATA.append((A, B))
        
    # 3. Campaign Loop
    # 3. Campaign Loop
    base_dir = f"fault_data_ver10_{layer_name}"
    ensure_dir(base_dir)
    ensure_dir(base_dir)
    ensure_dir("tmp_workers_os")
    
    total_files = 0
    start_time = time.time()
    
    for r in range(N):
        for c in range(N):
            pe_dir = f"{layer_name}_8x8_PE{r}_{c}"
            
            for tgt in ["WEIGHT", "INPUT", "PSUM"]:
                t_dir = tgt.lower()
                save_dir = os.path.join(base_dir, pe_dir, t_dir)
                ensure_dir(save_dir)
                
                # Load Map (Active Cycles)
                map_file = f"active_cycles_map_os_{layer_name}.json" 
                active_map = {}
                if os.path.exists(map_file):
                    with open(map_file, "r") as f:
                        active_map = json.load(f)
                
                valid_local_cycles = active_map.get(f"{r},{c}", [])
                if not valid_local_cycles:
                    print(f"Skipping PE[{r}][{c}] {tgt} (No Active Cycles)")
                    continue

                # Determine Target for this run
                pe_target_samples = target_samples
                if args.exhaustive:
                    pe_target_samples = len(valid_local_cycles)

                # Check existing
                exist = len([x for x in os.listdir(save_dir) if x.startswith("faulty_")])
                if exist >= pe_target_samples: 
                    print(f"Skipping PE[{r}][{c}] {tgt} (Found {exist}/{pe_target_samples})")
                    continue
                    
                print(f"Compiling PE[{r}][{c}] {tgt}...")
                binary_name = compile_simulation(r, c, TARGET_MAP[tgt])

                # Global Pool (From Map)
                # Map contains Global Cycles already (accumulated by Tile order in Profiler)
                global_pool = valid_local_cycles
                if not args.exhaustive:
                    random.shuffle(global_pool)
                # Else: keep sorted order from Map
                
                # Batch Processing
                pool_idx = 0
                while exist < pe_target_samples and pool_idx < len(global_pool):
                    needed = pe_target_samples - exist
                    batch_size = max(needed * 2, 16)
                    batch = global_pool[pool_idx : pool_idx + batch_size]
                    pool_idx += len(batch)
                    
                    if not batch: break
                    
                    pool_args = []
                    for g in batch:
                        t_idx = g // TILE_DURATION
                        if t_idx < len(TILES_DATA):
                            mata, matb = TILES_DATA[t_idx]
                            pool_args.append((g, r, c, TILE_DURATION, mata, matb, pe_dir, t_dir, save_dir, out_ch, num_tiles_n, binary_name))
                            
                    with multiprocessing.Pool(NUM_PROCESSES) as pool:
                        results = pool.map(worker_simulation, pool_args)
                    
                    if os.path.exists(binary_name):
                        os.remove(binary_name)
                        
                    for _, created in results:
                        if created: exist += 1
                        
                    # Fix Overshoot: Delete excess files
                    if exist > pe_target_samples:
                        files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith("faulty_")]
                        files.sort(key=os.path.getmtime) # Sort by time (oldest first)
                        # We want to keep target_samples, so delete the newest (or oldest? doesn't matter much)
                        # Let's delete the NEWEST ones (extras)
                        excess = exist - pe_target_samples
                        files_to_delete = files[-excess:] 
                        for fpath in files_to_delete:
                            try:
                                os.remove(fpath)
                                exist -= 1
                            except OSError:
                                pass

                    print(f"   -> Progress: {exist}/{pe_target_samples}")
                    total_files += exist
                    
    print(f"Done. Total Files: {total_files}. Time: {time.time()-start_time:.1f}s")
    shutil.rmtree("tmp_workers_os", ignore_errors=True)

if __name__ == "__main__":
    main()

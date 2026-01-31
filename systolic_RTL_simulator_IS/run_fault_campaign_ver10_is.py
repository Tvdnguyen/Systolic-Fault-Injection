import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import subprocess
import os
import sys
import argparse
import re
import glob
import shutil
import csv
import random
import multiprocessing
import time
import json
from collections import defaultdict

# Import Model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) 
try:
    from traffic_sign_net_small import TrafficSignNetSmallNoDropout
except ImportError:
    if os.path.exists("traffic_sign_net_small.py"):
        from traffic_sign_net_small import TrafficSignNetSmallNoDropout
    else:
        # Fallback Mock
        class TrafficSignNetSmallNoDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, 1)

# ==============================================================================
# CONFIG
# ==============================================================================
DATAFLOW = "IS"
N = 8    
L_SCAN = 100 
NUM_PROCESSES = os.cpu_count() or 8

TARGET_MAP = {
    "WEIGHT": 0, # Stream
    "INPUT": 1,  # Stationary
    "PSUM": 2
}

def ensure_dir(path):
    if not os.path.exists(path):
        try: os.makedirs(path)
        except: pass

# ==============================================================================
# DATA GENERATION
# ==============================================================================
# IS: act.mem = Stationary Inputs (mat_a). weight.mem = Stream Weights (mat_b).
def generate_mem_files(mat_a, mat_b, prefix=""):
    # mat_a: Inputs (Stationary Tile N*N)
    # mat_b: Weights (Stream Tile N*L)
    
    a_name = f"{prefix}act.mem"
    w_name = f"{prefix}weight.mem"
    
    # Write Stationary Inputs (act.mem) -> flattened Row-Major
    with open(a_name, "w") as f:
         flat_a = mat_a.flatten().numpy().astype(int)
         for val in flat_a:
             f.write(f"{val:02x}\n")
             
    # ---------------------------------------------------------
    # 2. GENERATE weight.mem (STREAMING - HORIZONTAL - SKEWED)
    # ---------------------------------------------------------
    # Matrix B is (Rows, Cols) = (N, L_SCAN).
    # We stream it into Left Port (m0).
    # Cross-Flow Skew: Row r is delayed by r cycles.
    # Buffer Size: Time = L_SCAN + N. Space = N.
    
    L_TIME = L_SCAN
    input_buffer = np.zeros((L_TIME + N, N), dtype=int)
    mat_b_np = mat_b.numpy().astype(int)
    
    # Fill Buffer with Skew
    for r in range(N):
        delay = r
        for t in range(L_TIME):
             input_buffer[t + delay, r] = mat_b_np[r, t]
             
    with open(w_name, "w") as f:
         # Write Buffer (Time Priority)
         for t in range(L_TIME + N):
             for r in range(N): # Drive Left Ports m0[r]
                 val = input_buffer[t, r]
                 f.write(f"{val:02x}\n")
             
    return w_name, a_name


# ==============================================================================
# SIMULATION WORKER
# ==============================================================================
def compile_simulation(f_row, f_col, f_target):
    cmd = [
        "iverilog", "-g2012",
        f"-P", f"dut_tb_is.FAULT_ROW={f_row}",
        f"-P", f"dut_tb_is.FAULT_COL={f_col}",
        f"-P", f"dut_tb_is.FAULT_TARGET={f_target}",
        "-o", "sim_campaign_is",
        "dut_tb_is.sv", 
        "systolic_is.sv",
        "systolic_is_faulty.sv",
        "pe_is.v", 
        "pe_is_faulty.v",
        "pe_control_is.v",
        "pe_datapath_is.v",
        "pe_datapath_is_faulty.v" 
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def worker_simulation(args):
    g_cyc, r, c, tile_duration, mat_a, mat_b, pe_dir_name, t_dir_name, save_dir, out_ch = args
    job_id = f"{os.getpid()}_{g_cyc}"
    
    # Unique I/O Files
    weight_file = os.path.join("tmp_workers", f"weight_{job_id}.mem")
    act_file = os.path.join("tmp_workers", f"act_{job_id}.mem")
    output_file = os.path.join("tmp_workers", f"output_{job_id}.txt")
    
    # Logic Update: Using updated generate_mem_files logic inline
    # mat_a (Inputs Stat), mat_b (Weights Stream)
    try:
        with open(act_file, "w") as f:
             flat = mat_a.flatten().numpy().astype(int)
             for v in flat: f.write(f"{v:02x}\n")
             
        with open(weight_file, "w") as f:
             # Stream Weights (mat_b). Needed format: [Time (L), N]
             # mat_b is [N, L]. Transpose to [L, N] and flatten.
             flat = mat_b.t().flatten().numpy().astype(int)
             for v in flat: f.write(f"{v:02x}\n")
        
        # Determine Local Cycle
        # Logic: g_cyc is absolute.
        # local_cyc = g_cyc % tile_duration
        local_cyc = g_cyc % tile_duration
        max_cyc = tile_duration + 50 
        
        # Run VVP
        cmd_run = [
            "vvp", "sim_campaign_is",
            f"+FAULT_ENABLE=1", 
            f"+FAULT_ROW={r}",
            f"+FAULT_COL={c}",
            f"+FAULT_CYCLE={local_cyc}",
            f"+FAULT_VAL=1",
            f"+MAX_CYCLES={max_cyc}",
            f"+OUTPUT_FILE={output_file}",
            f"+MEM_WEIGHT={weight_file}",
            f"+MEM_ACT={act_file}"
        ]
        
        subprocess.run(cmd_run, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        gold_data, fault_data = [], []
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    raw_parts = line.strip().split(',')
                    if len(raw_parts) < 3: continue
                    data_numeric = [int(x) for x in raw_parts[1:] if x.strip().isdigit() or x.strip().replace('-','').isdigit()]
                    if len(data_numeric) >= 2*N:
                        gold_data.append(data_numeric[0:N])
                        fault_data.append(data_numeric[N:2*N])
                        
        gold = np.array(gold_data)
        faulty = np.array(fault_data)
        
        diff_idx_t, diff_idx_ch = [], []
        if len(gold) > 0 and len(faulty) > 0:
            min_len = min(len(gold), len(faulty))
            gold = gold[:min_len]
            faulty = faulty[:min_len]
            diff = np.abs(gold - faulty)
            if np.max(diff) > 0:
                locs = np.where(diff > 0.1)
                diff_idx_t, diff_idx_ch = locs[0], locs[1]

        if len(diff_idx_t) > 0:
            valid_lines = []
            for i in range(len(diff_idx_t)):
                ch_local = diff_idx_ch[i]
                flat_idx = diff_idx_t[i]
                valid_lines.append(f"{flat_idx},{ch_local}") 
            return (g_cyc, valid_lines)

    except Exception as e:
        return (g_cyc, [])
    finally:
        # Cleanup
        if os.path.exists(weight_file): os.remove(weight_file)
        if os.path.exists(act_file): os.remove(act_file)
        if os.path.exists(output_file): os.remove(output_file)
        
    return (g_cyc, [])

# ==============================================================================
# MAIN
# ==============================================================================
def calculate_active_cycles_is(N, L_SCAN, ROW, COL):
    RESET_CYCLES = 2
    LOAD_PHASE = N
    OFFSET_TB = RESET_CYCLES + LOAD_PHASE
    start_cycle = OFFSET_TB + ROW + COL 
    end_cycle = OFFSET_TB + ROW + COL + L_SCAN - 1
    return start_cycle, end_cycle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=str, default="conv1")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--exhaustive", action="store_true") # Restored Exhaustive
    args = parser.parse_args()
    
    layer_name = args.layer
    base_target = args.samples
    
    # Fix UnboundLocalError: Initialize early.
    # If exhaustive, we set it artificially high to pass the initial bucket checks,
    # then reset it to the exact map size later.
    if args.exhaustive:
        target_per_bucket = 999999999
    else:
        target_per_bucket = base_target
    
    print(f"=== VER9_4: OPTIMIZED PARALLEL ({DATAFLOW}) ===")
    
    LAYER_DIMS = {
        'conv1': (32, 32), 'conv2': (32, 32),
        'conv3': (16, 16), 'conv4': (16, 16),
        'conv5': (8, 8),   'conv6': (8, 8)
    }
    H_out, W_out = LAYER_DIMS.get(layer_name, (32, 32))
    
    model = TrafficSignNetSmallNoDropout()
    try: layer_obj = dict(model.named_modules())[layer_name]
    except: layer_obj = model.conv1
    
    out_ch = layer_obj.out_channels
    M_total = out_ch
    N_total = H_out * W_out
    
    OFFSET = 18 # Aligned with WS (Load Phase Included in Dump)
    # Conv1: 3x3x3=27 -> 32 (4 chunks)
    # Conv3: 16x3x3=144 -> 144 (18 chunks)
    # Conv5: 16x3x3=144? No wait, Check model.
    # TrafficSignNetSmall:
    # Conv1: In=3, Out=16, K=3. Depth=3*9=27. Pad to 32. Chunks=4.
    # Conv3: In=16, Out=16, K=3. Depth=16*9=144. Pad to 144. Chunks=18.
    # Conv5: In=32 (From Pool2?), Out=32, K=3.
    # Actually, let's rely on the predefined map counts in generate_maps_verified_64pe_global.py
    
    num_tiles_m = (M_total + N - 1) // N
    num_tiles_n = (N_total + N - 1) // N
    
    # Map-Specific Total Ops (Must match map generation)
    TOTAL_OPS_MAP = {
        'conv1': 1024,
        'conv3': 2304,
        'conv5': 2304
    }
    K_CHUNKS_MAP = {
        'conv1': 4,
        'conv3': 18,
        'conv5': 36
    }
    total_tiles = TOTAL_OPS_MAP.get(layer_name, 256)
    k_chunks = K_CHUNKS_MAP.get(layer_name, 1)
    
    LOAD_TIME = N 
    COMPUTE_TIME = L_SCAN 
    DRAIN_TIME = 2 * N 
    TILE_DURATION = LOAD_TIME + COMPUTE_TIME + DRAIN_TIME
    
    TILES_DATA = []
    # Generate random data for ALL atomic operations
    for t_idx in range(total_tiles):
        # IS: mat_a = Inputs (Stationary Tile N*N)
        # IS: mat_b = Weights (Stream Tile N*L)
        mat_a = torch.randint(1, 127, (N, N))
        mat_b = torch.randint(1, 127, (N, L_SCAN))
        TILES_DATA.append((mat_a, mat_b))
        
    start_time = time.time()
    total_files = 0
    base_dir = f"fault_data_ver9_4_{layer_name}"
    ensure_dir(base_dir)
    
    if os.path.exists("tmp_workers"): shutil.rmtree("tmp_workers")
    os.makedirs("tmp_workers")

    for r in range(8):
        for c in range(8):
            pe_dir_name = f"{layer_name}_8x8_PE{r}_{c}"
            
            for tgt_name in ["WEIGHT", "INPUT", "PSUM"]:
                t_dir_name = tgt_name.lower()
                save_dir = os.path.join(base_dir, pe_dir_name, t_dir_name)
                ensure_dir(save_dir)
                
                existing_files = [name for name in os.listdir(save_dir) if name.startswith("faulty_")]
                files_in_bucket = len(existing_files)
                if files_in_bucket >= target_per_bucket: continue
                
                tgt_id = TARGET_MAP[tgt_name]
                print(f"Compiling PE[{r}][{c}] {tgt_name}...")
                compile_simulation(r, c, tgt_id)
                
                # User will manually copy WS maps to IS filenames.
                # Logic inside file is WS-compatible (Global Cycles).
                map_file = f"active_cycles_map_is_{layer_name}.json"
                
                active_map = {}
                if os.path.exists(map_file):
                    with open(map_file, "r") as f:
                        active_map = json.load(f)
                    print(f"Loaded Active Cycles Map (IS Filename, WS Logic): {map_file}")
                
                valid_local_cycles = active_map.get(f"{r}_{c}", [])
                
                # If map missing, fallback to logic
                if not valid_local_cycles:
                     local_start, local_end = calculate_active_cycles_is(N, L_SCAN, r, c)
                     valid_local_cycles = list(range(local_start, local_end + 1))
                
                global_pool = []
                
                # Global Map Support Check
                if len(valid_local_cycles) > 200:
                    global_pool = valid_local_cycles
                else:
                    for t_idx in range(total_tiles):
                        tile_offset = t_idx * TILE_DURATION
                        for lc in valid_local_cycles:
                            global_pool.append(tile_offset + lc)
                
                if not args.exhaustive:
                    random.shuffle(global_pool)
                
                # Update Target if Exhaustive
                if args.exhaustive:
                    target_per_bucket = len(global_pool)
                    possible_target = len(global_pool)
                else:
                    target_per_bucket = base_target
                    possible_target = base_target

                pool_idx = 0
                max_pool = len(global_pool)
                
                while files_in_bucket < possible_target and pool_idx < max_pool:
                    needed = possible_target - files_in_bucket
                    batch_size = max(needed * 3, 20)
                    
                    candidates = global_pool[pool_idx : min(pool_idx + batch_size, max_pool)]
                    pool_idx += len(candidates)
                    
                    if not candidates: break
                    
                    pool_args = []
                    for g_cyc in candidates:
                        t_idx = g_cyc // TILE_DURATION
                        if t_idx < len(TILES_DATA):
                            mat_a, mat_b = TILES_DATA[t_idx]
                            pool_args.append((g_cyc, r, c, TILE_DURATION, mat_a, mat_b, pe_dir_name, t_dir_name, save_dir, out_ch))
                    
                    print(f"   PE[{r}][{c}] {tgt_name}: Spawning batch of {len(pool_args)} tasks...")
                    
                    with multiprocessing.Pool(NUM_PROCESSES) as pool:
                        results = pool.map(worker_simulation, pool_args)
                    
                    # Process Results
                    for g_cyc, valid_lines_raw in results:
                        if files_in_bucket >= possible_target: break
                        if len(valid_lines_raw) > 0:
                            final_lines = []
                            t_idx = g_cyc // TILE_DURATION
                            
                            # Handle K-Dimension flattening
                            spatial_t_idx = t_idx // k_chunks
                            
                            tile_m_idx = spatial_t_idx // num_tiles_n
                            tile_n_idx = spatial_t_idx % num_tiles_n
                            
                            for line in valid_lines_raw:
                                flat_idx, ch_local = map(int, line.split(','))
                                # IS CROSS-FLOW MAPPING
                                # Space (Col) -> Pixel
                                # Time (Cycle) -> OutCh (Filter) with Skew
                                # Output Wavefront is Diagonal: Result(k, c) arrives at T = k + c + OFFSET
                                # So: k (g_ch) = T - c - OFFSET
                                
                                # 1. Global Pixel
                                # g_px = col_offset + local_pixel
                                # local_pixel is local_col (0..7)
                                g_px = ch_local # Simplified for single tile or needs offset?
                                # To be safe: g_px = col_idx_global? 
                                # Actually g_px usually relates to Tile Col.
                                # But here we just want local mapping.
                                g_px = ch_local # This is 'p'
                                
                                # 2. Global Channel (OutCh)
                                # local_filter (k) = flat_idx (Cycle) - g_px - OFFSET
                                # Note: flat_idx here is the Cycle number from file (after removing offset in loop?)
                                # Logic: `flat_idx` in loop below is `cycle`.
                                # So:
                                local_filter = flat_idx - g_px - OFFSET
                                
                                # Map to Global Dimensions
                                # We assume standard NCHW or similar.
                                g_ch = local_filter
                                
                                # Row/Col Image from Pixel?
                                # If g_px is linear pixel index.
                                g_row = g_px // 32 # Assuming Width 32
                                g_col_img = g_px % 32
                                
                                if local_filter >= 0 and local_filter < 100: # L_SCAN=100 (Standard WS)
                                    # Global Mapping
                                    # g_ch = tile_m_idx * N + local_filter 
                                    g_ch = tile_m_idx * N + local_filter 
                                    g_px = tile_n_idx * N + g_px 
                                    
                                    g_row = g_px // 32
                                    g_col_img = g_px % 32
                                    final_lines.append(f"{g_ch}, {g_row}, {g_col_img}\n")
                                
                                # Removed incorrect else block that forced local_row = r
                                # This ensures temporal shifts in faults (cycle changes) are not clamped to a constant.

                            if len(final_lines) > 0:
                                fname = f"faulty_{pe_dir_name}_{t_dir_name}_cycle_{g_cyc}.txt"
                                fpath = os.path.join(save_dir, fname)
                                with open(fpath, "w") as f_out:
                                    for l in final_lines: f_out.write(l)
                                files_in_bucket += 1
                                total_files += 1
                                
                print(f"   PE[{r}][{c}] {tgt_name}: Finished. Total: {files_in_bucket}/{possible_target}")

    print(f"Total Time: {time.time() - start_time:.2f}s")
    if os.path.exists("tmp_workers"): shutil.rmtree("tmp_workers")

if __name__ == "__main__":
    main()

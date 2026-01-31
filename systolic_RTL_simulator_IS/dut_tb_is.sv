module dut_tb_is #(
    parameter D_W = 8,
    parameter N = 8,
    parameter M = 8,
    parameter FAULT_ROW = 0,
    parameter FAULT_COL = 0,
    parameter FAULT_TARGET = 0 // 0:Weight, 1:Input
)();

    reg clk = 0;
    reg rst = 1;
    reg load_weight = 0;
    
    // Arrays for IO (Shared Inputs)
    reg [D_W-1:0] m0 [N-1:0]; // Left (Acts)
    reg [D_W-1:0] m1 [N-1:0]; // Top (Weights)
    
    // Outputs
    wire [2*D_W-1:0] m2_gold [N-1:0];
    wire [2*D_W-1:0] m2_fault [N-1:0];
    
    // Fault Masks for Faulty DUT
    reg [D_W-1:0] fault_masks [N-1:0][N-1:0];
    
    // Memory
    reg [D_W-1:0] mem_weights [0:(1024*N)-1]; // Large stream for IS
    reg [D_W-1:0] mem_acts    [0:(N*N)-1];    // Small tile for IS Load
    
    // Simulation Control
    integer cycle_count = 0;
    integer phase = 0;
    integer max_cycles = 2000;
    string output_file = "output_dual.txt";
    
    // Fault Params
    integer fault_enable = 0;
    integer fault_row = 0;
    integer fault_col = 0;
    integer fault_cycle = 0;
    integer fault_val = 0; 
    
    initial begin
        if ($value$plusargs("FAULT_ENABLE=%d", fault_enable)) begin
            $value$plusargs("FAULT_ROW=%d", fault_row);
            $value$plusargs("FAULT_COL=%d", fault_col);
            $value$plusargs("FAULT_CYCLE=%d", fault_cycle);
            $value$plusargs("FAULT_VAL=%d", fault_val);
        end
        $value$plusargs("MAX_CYCLES=%d", max_cycles);
        $value$plusargs("OUTPUT_FILE=%s", output_file);
    end
    
    // Instantiate GOLDEN
    // Instantiate GOLDEN
    systolic_is #(.D_W(D_W), .N(N), .M(M)) dut_gold (
        .clk(clk),
        .rst(rst),
        .load_weight(load_weight),
        .m0(m0),
        .m1(m1),
        .m2(m2_gold)
    );
    
    // Instantiate FAULTY
    systolic_is_faulty #(.D_W(D_W), .N(N), .M(M), .FAULT_ROW(FAULT_ROW), .FAULT_COL(FAULT_COL), .FAULT_TARGET(FAULT_TARGET)) dut_faulty (
        .clk(clk),
        .rst(rst),
        .load_weight(load_weight),
        .m0(m0),
        .m1(m1),
        .fault_masks(fault_masks),
        .m2(m2_fault)
    );
    
    // Clock
    always #5 clk = ~clk;
    
    // Cycle Counter
    always @(posedge clk) begin
        if (!rst) cycle_count <= cycle_count + 1;
    end
    
    // Initialization
    integer r, c, t;
    initial begin
        // OPTIMIZATION: Disable VCD
        // $dumpfile("dual_wave.vcd");
        // $dumpvars(0, dut_tb_is);
        
        string mem_w_file = "weight.mem";
        string mem_a_file = "act.mem";
        if ($value$plusargs("MEM_WEIGHT=%s", mem_w_file));
        if ($value$plusargs("MEM_ACT=%s", mem_a_file));
        
        $readmemh(mem_w_file, mem_weights);
        $readmemh(mem_a_file, mem_acts);
        
        // Reset
        rst = 1;
        load_weight = 0;
        for(r=0; r<N; r=r+1) m0[r] = 0;
        for(c=0; c<N; c=c+1) m1[c] = 0;
        
        // Init Fault Masks to 0 (Clean)
        for(r=0; r<N; r=r+1) begin
            for(c=0; c<N; c=c+1) begin
                fault_masks[r][c] = 0;
            end
        end
        
        #20;
        rst = 0;
        $display("RESET_TIME: %0t", $time);
        
        // --- Phase 1: Load Inputs (Stationary) ---
        $display("Starting Phase 1: Load Inputs (Vertical Fill)");
        load_weight = 1; // Used as LOAD_ENABLE
        phase = 1;
        
        // Feed Rows in Reverse (N-1 down to 0) to fill array Top-to-Bottom
        for (r = N-1; r >= 0; r = r - 1) begin
            @(posedge clk);
            #1;
            for (c = 0; c < N; c = c + 1) begin
                m1[c] = mem_acts[r*N + c]; // Drive Top Port with Input Matrix Row
            end
        end
        
        @(posedge clk);
        #1;
        load_weight = 0;
        
        // --- Phase 2: Compute (Stream Weights Horizontal) ---
        $display("Starting Phase 2: Compute (Stream Weights Horizontal)");
        `ifdef PROFILING
        $display("PHASE2_START,%0d", $time);
        `endif
        phase = 2;
        for(c=0; c<N; c=c+1) m1[c] = 0; // Quiet Inputs
        
        // OPTIMIZATION: Dynamic Cycle Limit
        for (t = 0; t < max_cycles; t = t + 1) begin
            // Fault Injection Trigger (Transient 1 Cycle)
            if (fault_enable && cycle_count == fault_cycle) begin
                 // $display("Injecting Transient Fault at Cycle %d on PE[%d][%d]", cycle_count, fault_row, fault_col);
                 fault_masks[fault_row][fault_col] = fault_val;
            end else begin
                 // Reset Mask (Transient)
                 fault_masks[fault_row][fault_col] = 0;
            end
            
            @(posedge clk);
            #1;
            for (r = 0; r < N; r = r + 1) begin
                m0[r] = mem_weights[t*N + r]; // Drive Left Port with Weight Stream
                if (m0[r] === 8'bx) m0[r] = 0;
            end
        end
        
        $finish;
    end
    
    // Logging Logic
    integer f;
    initial begin
        f = $fopen(output_file, "w");
    end
    
    always @(posedge clk) begin
        if (phase == 2) begin
            // Log Cycle, then Gold 0..N-1, then Faulty 0..N-1
            $fwrite(f, "%d", cycle_count);
            for (r = 0; r < N; r = r + 1) $fwrite(f, ",%d", m2_gold[r]);
            for (r = 0; r < N; r = r + 1) $fwrite(f, ",%d", m2_fault[r]);
            $fwrite(f, "\n");
        end
    end

endmodule

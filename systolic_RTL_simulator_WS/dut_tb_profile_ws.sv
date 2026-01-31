module dut_tb_profile_ws #(
    parameter D_W = 8,
    parameter N = 8,
    parameter M = 8
)();

    reg clk = 0;
    reg rst = 1;
    reg load_weight = 0;
    
    // Arrays for IO (Shared Inputs)
    reg [D_W-1:0] m0 [N-1:0]; // Left (Acts)
    reg [D_W-1:0] m1 [N-1:0]; // Top (Weights)
    
    // Outputs
    wire [2*D_W-1:0] m2_gold [N-1:0];
    
    // Memory
    reg [D_W-1:0] mem_weights [0:(N*N)-1];
    reg [D_W-1:0] mem_acts    [0:(1024*N)-1];
    
    // Simulation Control
    integer cycle_count = 0;
    integer phase = 0;
    integer max_cycles = 2048; 
    string profile_file = "profile_log_ws.txt";

    // Instantiate GOLDEN (IS)
    systolic_ws #(.D_W(D_W), .N(N), .M(M)) dut_gold (
        .clk(clk),
        .rst(rst),
        .load_weight(load_weight),
        .m0(m0),
        .m1(m1),
        .m2(m2_gold)
    );
    
    // Clock
    always #5 clk = ~clk;
    
    always @(posedge clk) begin
        if (!rst) cycle_count <= cycle_count + 1;
    end
    
    // =========================================================================
    // PROFILING LOGIC
    // =========================================================================
    reg [0:0] pe_active [N-1:0][N-1:0];
    genvar i, j;
    generate
        for (i=0; i<N; i=i+1) begin : mon_rows
            for (j=0; j<N; j=j+1) begin : mon_cols
                always @(*) begin
                     // IS: Input is horizontal (in_act), Weight is internal (weight_reg)
                     // Access: dut_gold.rows[i].cols[j].pe_inst.u_datapath...
                     pe_active[i][j] = (dut_gold.rows[i].cols[j].pe_inst.u_datapath.in_act != 0) && 
                                       (dut_gold.rows[i].cols[j].pe_inst.u_datapath.weight_reg != 0);
                end
            end
        end
    endgenerate

    integer f_log;
    initial begin
        f_log = $fopen(profile_file, "w");
    end

    integer r_idx, c_idx;
    always @(posedge clk) begin
        // Only log during Compute Phase and if Reset is inactive
        if (!rst && !load_weight && phase == 2) begin
             for (r_idx=0; r_idx<N; r_idx=r_idx+1) begin
                 for (c_idx=0; c_idx<N; c_idx=c_idx+1) begin
                     if (pe_active[r_idx][c_idx]) begin
                         // Format: Cycle, Row, Col
                         $fwrite(f_log, "%d,%d,%d\n", cycle_count, r_idx, c_idx);
                     end
                 end
             end
        end
    end

    // =========================================================================
    // DRIVER (Similar to WS for IS in this verification setup)
    // =========================================================================
    integer r, c, t;
    initial begin
        $readmemh("weight.mem", mem_weights);
        $readmemh("act.mem", mem_acts);
        
        rst = 1;
        load_weight = 0;
        for(r=0; r<N; r=r+1) m0[r] = 0;
        for(c=0; c<N; c=c+1) m1[c] = 0;
        
        #20;
        rst = 0;
        
        // --- Phase 1: Load Weights ---
        load_weight = 1;
        phase = 1;
        
        // IS Load Logic: Usually loads weights into PEs
        // The script generates weight.mem assuming WS loading order?
        // Let's assume standard systolic loading from Top for IS too (depends on RTL)
        // RTL check: pe_ws accepts in_weight from Top and stores it if load_weight=1.
        // Similar to WS.
        for (r = N-1; r >= 0; r = r - 1) begin
            @(posedge clk);
            #1;
            for (c = 0; c < N; c = c + 1) begin
                m1[c] = mem_weights[r*N + c];
            end
        end
        
        @(posedge clk);
        #1;
        load_weight = 0;
        
        // --- Phase 2: Compute ---
        phase = 2;
        for(c=0; c<N; c=c+1) m1[c] = 0;
        
        // Feed Inputs
        for (t = 0; t < max_cycles; t = t + 1) begin
            @(posedge clk);
            #1;
            for (r = 0; r < N; r = r + 1) begin
                m0[r] = mem_acts[t*N + r];
                if (m0[r] === 8'bx) m0[r] = 0;
            end
        end
        
        $finish;
    end

endmodule

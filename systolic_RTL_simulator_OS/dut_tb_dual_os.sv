module dut_tb_dual_os
#(
    parameter D_W = 8,
    parameter N = 5,
    parameter M = 5,
    parameter FAULT_ROW = 0,
    parameter FAULT_COL = 0,
    parameter FAULT_TARGET = 0 // 0:W, 1:I, 2:P
)
();

	reg                                 clk=1'b0;
	reg     [1:0]                       rst;

	// Shared Memory Reader Logic
	reg                              rd_en_m0=0;
	reg                              rd_en_m1=0;
	wire    [N-1:0]                  rd_en_m0_pipe;
	wire    [N-1:0]                  rd_en_m1_pipe;
	wire    [$clog2((M*M)/N)-1:0]    rd_addr_m0 [N-1:0];
	wire    [$clog2((M*M)/N)-1:0]    rd_addr_m1 [N-1:0];
	wire    [$clog2(M)-1:0]          column_m0;
	wire    [$clog2(M/N)-1:0]        row_m0;
	wire    [$clog2(M/N)-1:0]        column_m1;
	wire    [$clog2(M)-1:0]          row_m1;
	
	// Data Pipes (Shared Input to DUTs)
	reg     [D_W-1:0]                m0_pipe    [N-1:0];
	reg     [D_W-1:0]                m1_pipe    [N-1:0];

	// DUT Outputs
	wire    [2*D_W-1:0]              m2_gold    [N-1:0];
	wire    [N-1:0]                  valid_gold;
	
	wire    [2*D_W-1:0]              m2_fault   [N-1:0];
	wire    [N-1:0]                  valid_fault;
	
	// Fault Masks
	reg     [D_W-1:0]                fault_masks [N-1:0][N-1:0];

	// Memory Storage
	reg     [D_W-1:0]                mem0       [0:(M*M)-1];
	reg     [D_W-1:0]                mem1       [0:(M*M)-1];
	
	// Simulation Control
	integer cycle_count = 0;
	integer fault_enable = 0;
    integer fault_row_rt = 0; // Runtime args (masked by Param)
    integer fault_col_rt = 0;
    integer fault_cycle = 0;
    integer fault_val = 0;
    integer max_cycles = 5000; // Default
    string output_file = "output_dual_os.txt";

	initial begin
		$readmemh("m0.mem", mem0);
		$readmemh("m0.mem", mem0);
		$readmemh("m1.mem", mem1);
		
		if ($value$plusargs("FAULT_ENABLE=%d", fault_enable)) begin
             $value$plusargs("FAULT_ROW=%d", fault_row_rt);
             $value$plusargs("FAULT_COL=%d", fault_col_rt);
             $value$plusargs("FAULT_CYCLE=%d", fault_cycle);
             $value$plusargs("FAULT_VAL=%d", fault_val);
         end
         $value$plusargs("MAX_CYCLES=%d", max_cycles);
         $value$plusargs("OUTPUT_FILE=%s", output_file);
         
         // Open file AFTER parsing args
         f = $fopen(output_file, "w");
	end

	always@(posedge clk) begin
		if (rst[0]) begin
			rd_en_m0 <= 0;
			rd_en_m1 <= 0;
		end else begin
			rd_en_m0 <= 1;
			rd_en_m1 <= 1;
		end
	end
	
	// Cycle Counter
	always @(posedge clk) begin
	    if (!rst[0]) cycle_count <= cycle_count + 1;
	end

    // --- INSTANTIATE READERS ---
	mem_read_m0 #(.D_W(D_W), .N(N), .M(M)) mem_read_m0_inst (
		.clk           (clk),
		.row            (row_m0),
		.column         (column_m0),
		.rd_en          (~rst[0]),
		.rd_addr_bram   (rd_addr_m0),
		.rd_en_bram     (rd_en_m0_pipe)     
	);
	
	mem_read_m1 #(.D_W(D_W), .N(N), .M(M)) mem_read_m1_inst (
		.clk           (clk),
		.row            (row_m1),
		.column         (column_m1),
		.rd_en          (~rst[0]),
		.rd_addr_bram   (rd_addr_m1),
		.rd_en_bram     (rd_en_m1_pipe)     
	);

    // --- DATA FETCH LOGIC ---
	genvar x;
	for(x=0;x<N;x=x+1) begin
		always@(posedge clk) begin
			if (rst[0]==1'b0) begin
				if (rd_en_m0_pipe[x]) m0_pipe[x] <= mem0[((x*M*M)/N) + rd_addr_m0[x]];
				else                  m0_pipe[x] <= 0;
			end else m0_pipe[x] <= 0;
		end
		
		always@(posedge clk) begin
			if (rst[0]==1'b0) begin
				if (rd_en_m1_pipe[x]) m1_pipe[x] <= mem1[((x*M*M)/N) + rd_addr_m1[x]];
				else                  m1_pipe[x] <= 0;
			end else m1_pipe[x] <= 0;
		end
	end

    // --- SHARED CONTROL SIGNALS ---
	reg enable_row_count_m0 = 0;
	reg [31:0]  patch =1;
	
    // --- DUT 1: GOLDEN ---
	systolic #(.D_W(D_W), .N(N), .M(M)) dut_gold (
		.clk            (clk), 
		.rst            (rst[0]),
		.enable_row_count_m0    (enable_row_count_m0),
		.column_m0      (column_m0),
		.column_m1      (column_m1),
		.row_m0         (row_m0),
		.row_m1         (row_m1),
		.m0             (m0_pipe), 
		.m1             (m1_pipe),     
		.m2             (m2_gold),
		.valid_m2       (valid_gold)  
	);
	
	// --- DUT 2: FAULTY ---
	systolic_os_faulty #(
	    .D_W(D_W), .N(N), .M(M), 
	    .FAULT_ROW(FAULT_ROW), 
	    .FAULT_COL(FAULT_COL), 
	    .FAULT_TARGET(FAULT_TARGET)
	) dut_faulty (
		.clk            (clk), 
		.rst            (rst[0]),
		.enable_row_count_m0    (enable_row_count_m0),
        // Disconnect Counter Outputs to avoid contention with Gold
		.column_m0      (),
		.column_m1      (),
		.row_m0         (),
		.row_m1         (),
		.m0             (m0_pipe), 
		.m1             (m1_pipe),     

		.fault_masks    (fault_masks),
		.m2             (m2_fault),
		.valid_m2       (valid_fault)
		// We assume counters inside Faulty running identically.
	);

	always #0.5 clk = ~clk;

	initial begin
		rst = 2'b11;
		// Init Masks
		for(integer r=0; r<N; r=r+1)
		    for(integer c=0; c<N; c=c+1)
		        fault_masks[r][c] = 0;
	end

	always @(posedge clk) rst <= rst>>1;

    // --- CONTROL LOGIC (Copied from original TB to drive Sim) ---
    // Needed to toggle 'enable_row_count_m0' etc.
    // This logic depends on 'column_m0' from Gold DUT.
	always@(posedge clk) begin
		if(rst[0]) begin
			enable_row_count_m0 <= 1'b0;
		end else begin
            // Force Continuous Enable for OS Debug
            enable_row_count_m0 <= 1'b1;
		end
	end
	
	// --- FAULT INJECTION CONTROLLER ---
	always @(posedge clk) begin
	   if (fault_enable && cycle_count == fault_cycle) begin
	       // Check if param matches runtime target (Sanity check)
	       if (FAULT_ROW == fault_row_rt && FAULT_COL == fault_col_rt) begin
	           $display("Injecting OS Fault at Cycle %d on PE[%d][%d]", cycle_count, FAULT_ROW, FAULT_COL);
	           fault_masks[FAULT_ROW][FAULT_COL] <= fault_val;
	       end
	   end else begin
	       fault_masks[FAULT_ROW][FAULT_COL] <= 0; // Transient
	   end
	end

    // --- LOGGER ---
    integer f;
    initial begin
        // Defer opening until arguments are parsed
    end
    
    // We log every cycle where VALID is High OR if we want full trace?
    // WS logged phases. OS is bursty.
    // Let's log valid cycles.
    // Gold and Faulty SHOULD have same valid timing (unless Control Fault).
    // Logic: If ANY valid_gold is high, log line.
    
    integer r;
    always @(posedge clk) begin
        #1; // Delay to capture registered outputs
        if (valid_gold !== 0 || valid_fault !== 0) begin
            $fwrite(f, "%d", cycle_count);
            // Valid vectors
             $fwrite(f, ",%h,%h", valid_gold, valid_fault);
             
            // Outputs
            for (r = 0; r < N; r = r + 1) $fwrite(f, ",%d", m2_gold[r]);
            for (r = 0; r < N; r = r + 1) $fwrite(f, ",%d", m2_fault[r]);
            $fwrite(f, "\n");
        end
        // Debug Monitor
        if (cycle_count % 100 == 0) begin
            $display("Debug [%d]: Col=%d ValidG=%b Init00=%b", cycle_count, column_m0, valid_gold, dut_faulty.init[0][0]);
        end
    end
    
    // --- FINISH CONDITION ---
    // When Gold is done?
    // Original TB uses addr counter.
    // We can just timeout or count valid output patches.
    // Rough timeout is safest for Campaign.
    initial begin
        #50000; // Adjust as needed
        if (f) $fclose(f);
        $finish;
    end
    
    // Check finish from Gold Counter logic
	reg [$clog2((M*M)/N):0]   addr_check    [N-1:0];
	for (x=0;x<N;x=x+1) begin
		always@(posedge clk) begin
			if (rst[0]==1'b1) addr_check[x] <= 0;
			else if (valid_gold[x]) addr_check[x] <= addr_check[x] + 1;
		end
	end
	always@(posedge clk) begin
		if (addr_check[N-1]==((M*M)/N)) begin
			#100; // Drain
			if (f) $fclose(f);
			$finish;
		end
	end

endmodule

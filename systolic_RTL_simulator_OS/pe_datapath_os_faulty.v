module pe_datapath_os_faulty #(
    parameter D_W = 8,
    parameter FAULT_TARGET = 0, // 0:Weight(in_b), 1:Input(in_a), 2:Psum(out_tmp)
    parameter ROW = 0,
    parameter COL = 0
) (
    input  wire                clk,
    input  wire                rst,
    input  wire [D_W-1:0]      in_a,
    input  wire [D_W-1:0]      in_b,
    input  wire [2*D_W-1:0]    in_data,
    input  wire                in_valid,
    
    // Fault Interface
    input  wire [D_W-1:0]      fault_mask,

    input  wire                init_r,
    input  wire                in_valid_r,
    input  wire                data_rsrv,

    output wire [D_W-1:0]      out_a,
    output wire [D_W-1:0]      out_b,
    output reg  [2*D_W-1:0]    out_data,
    output wire                out_stagevalid_out
);

    // Inteernal Regs
    reg [D_W-1:0] a_tmp;
    reg [D_W-1:0] b_tmp;
    reg [2*D_W-1:0] in_data_r;
    reg [2*D_W-1:0] out_tmp;
    reg [2*D_W-1:0] out_tmp_r;
    reg [2*D_W-1:0] out_stage;
    reg out_stagevalid;

    // FAULT PROJECTION WIRES
    wire [D_W-1:0] effective_a;
    wire [D_W-1:0] effective_b;
    wire [2*D_W-1:0] effective_out_tmp;
    
    // FAULT APPLICATION LOGIC
    generate
        if (FAULT_TARGET == 0) begin : fault_weight_logic
            // Target WEIGHT (in_b) -> Vertical Flow & Compute
            assign effective_b = in_b ^ fault_mask;
            assign effective_a = in_a; 
        end else if (FAULT_TARGET == 1) begin : fault_input_logic
            // Target INPUT (in_a) -> Horizontal Flow & Compute
            assign effective_b = in_b;
            assign effective_a = in_a ^ fault_mask;
        end else begin : fault_psum_logic
            // Target PSUM (out_tmp) -> Accumulation
            assign effective_b = in_b;
            assign effective_a = in_a;
        end
    endgenerate

    // Multiplier (Uses Effective Operands - COMBINATIONAL)
    wire [2*D_W-1:0] mult_product;
    add_shift_multiplier_simple #(.D_W(D_W)) u_multiplier (
        .a(effective_a),
        .b(effective_b),
        .product(mult_product)
    );
    
    // Debug Trace
    always @(fault_mask) begin
        if (fault_mask != 0) begin
             if (FAULT_TARGET == 0) $display("OS Fault Trace: Mask %h -> Weight (B) | Val: %d -> %d", fault_mask, in_b, effective_b);
             if (FAULT_TARGET == 1) $display("OS Fault Trace: Mask %h -> Input (A) | Val: %d -> %d", fault_mask, in_a, effective_a);
             if (FAULT_TARGET == 2) $display("OS Fault Trace: Mask %h -> Psum Acc", fault_mask);
        end
    end
    
    // Profiling Trace
    `ifdef PROFILING
    always @(posedge clk) begin
       // Active if computing Non-Zero operands
       // Using effective values ensures we capture what truly enters the multiplier
       if (effective_a != 0 && effective_b != 0) begin
           $display("ACTIVE_LOG,%0d,%0d,%0d", ROW, COL, $time);
       end
    end
    `endif

    // --- Register: a_tmp (Input Horizon) ---
    // Captures EFFECTIVE (possibly faulty) value for propagation
    always @(posedge clk) begin
        if (rst) a_tmp <= 0;
        else     a_tmp <= effective_a;
    end

    // --- Register: b_tmp (Weight Vertical) ---
    // Captures EFFECTIVE (possibly faulty) value for propagation
    always @(posedge clk) begin
        if (rst) b_tmp <= 0;
        else     b_tmp <= effective_b;
    end

    // --- Register: in_data_r ---
    always @(posedge clk) begin
        if (rst) in_data_r <= 0;
        else     in_data_r <= in_data;
    end

    // --- Register: out_tmp_r (Mul Result) ---
    always @(posedge clk) begin
        if (rst) out_tmp_r <= 0;
        else     out_tmp_r <= mult_product;
    end

    // --- Register: out_tmp (Accumulator) ---
    always @(posedge clk) begin
        if (rst)
            out_tmp <= 0;
        else begin
            // Apply Fault to Accumulator if Target=2
            reg [2*D_W-1:0] next_val;
            
            case (init_r)
                1'b1: next_val = out_tmp_r;            
                1'b0: next_val = out_tmp + out_tmp_r;  
                default: next_val = out_tmp + out_tmp_r;
            endcase
            
            if (FAULT_TARGET == 2 && fault_mask != 0)
                out_tmp <= next_val ^ fault_mask;
            else
                out_tmp <= next_val;
        end
    end

    // --- Register: out_stage ---
    always @(posedge clk) begin
        if (rst) out_stage <= 0;
        else begin
            case ({init_r, in_valid_r, data_rsrv})
                3'b110, 3'b111, 3'b011: out_stage <= in_data_r; 
                3'b001, 3'b100, 3'b101, 3'b010, 3'b000: out_stage <= out_stage;
                default: out_stage <= out_stage;
            endcase
        end
    end

    // --- Register: out_stagevalid ---
    always @(posedge clk) begin
        if (rst) out_stagevalid <= 0;
        else begin
            case ({init_r, in_valid_r, data_rsrv})
                3'b110, 3'b111: out_stagevalid <= 1; // Force 1 when generating Psum
                3'b100, 3'b101: out_stagevalid <= 1;
                3'b011: out_stagevalid <= in_valid_r; // Shift neighbor valid
                default: out_stagevalid <= out_stagevalid;
            endcase
        end
    end

    // --- Register: out_data ---
    always @(posedge clk) begin
        if (rst) out_data <= 0;
        else begin
            case ({init_r, in_valid_r, data_rsrv})
                3'b110, 3'b111, 3'b100, 3'b101: out_data <= out_tmp; // Drain Accumulator
                3'b001, 3'b011: out_data <= out_stage;
                3'b010, 3'b000: out_data <= in_data_r;
                default: out_data <= in_data_r;
            endcase
        end
    end

    // --- Output Assignments (Propagate Effective Faults) ---
    // Always use registered values (a_tmp, b_tmp) for systolic propagation
    // a_tmp/b_tmp already capture 'effective_a'/'effective_b' logic.
    assign out_a = a_tmp;
    assign out_b = b_tmp;
    
    assign out_stagevalid_out = out_stagevalid;

endmodule

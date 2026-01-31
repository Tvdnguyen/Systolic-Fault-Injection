module pe_datapath_ws_faulty #(
    parameter D_W = 8,
    parameter FAULT_TARGET = 0, // 0:Weight(Stream), 1:Input(Stationary), 2:Psum(Stream)
    parameter ROW = 0,
    parameter COL = 0
)(
    input wire clk,
    input wire rst,
    input wire weight_we, 
    
    input wire [D_W-1:0] in_act,    // Stream (Horizontal)
    input wire [D_W-1:0] in_weight, // Stationary Load Source (Vertical)
    input wire [2*D_W-1:0] in_sum,
    
    output reg [D_W-1:0] out_act,
    output reg [D_W-1:0] out_weight,
    output reg [2*D_W-1:0] out_sum,
    
    // Fault Interface
    input wire [D_W-1:0] fault_mask
);

    reg [D_W-1:0] weight_reg; // STATIONARY DATA (INPUT) - Named weight_reg to match Golden
    wire [2*D_W-1:0] product;
    
    wire [D_W-1:0] effective_stream; // Corresponds to in_act
    wire [D_W-1:0] effective_stat;   // Corresponds to weight_reg
    
    // Fault Logic
    // Target 0: Stream (Weight in IS context, flowing Horizontally)
    // Target 1: Stationary (Input in IS context, Stored)
    // Target 2: Psum
    
    generate
        if (FAULT_TARGET == 0) begin 
            // Fault on Stream
            assign effective_stream = in_act ^ fault_mask;
            assign effective_stat = weight_reg;
        end else if (FAULT_TARGET == 1) begin
            // Fault on Stationary
            assign effective_stream = in_act;
            assign effective_stat = weight_reg ^ fault_mask;
        end else begin
            assign effective_stream = in_act;
            assign effective_stat = weight_reg;
        end
    endgenerate

    assign product = effective_stream * effective_stat;

    // PROBE LOGIC FOR Heuristic Script
    // PROBE LOGIC FOR Heuristic Script & Profiling
    always @(posedge clk) begin
         // Unconditional Log
         $display("Time %0d | PE[%0d][%0d] | In_Act: %d | Weight: %d | Product: %d | Sum: %d", $time, ROW, COL, in_act, weight_reg, product, out_sum);
    end

    always @(posedge clk) begin
        if (rst) begin
            weight_reg <= 0;
            out_act <= 0;
            out_weight <= 0;
            out_sum <= 0;
        end else begin
            // Pass-throughs
            out_weight <= in_weight; // Pass Vertical
            
            // Stream Pass-through (Faulty if Target 0)
            if (FAULT_TARGET == 0) out_act <= effective_stream;
            else out_act <= in_act;

            // Load Logic
            if (weight_we) begin
                weight_reg <= in_weight; // Vertical Load
                out_sum <= 0;
            end else begin
                // Compute
                if (FAULT_TARGET == 2)
                    out_sum <= (in_sum + product) ^ fault_mask;
                else
                    out_sum <= in_sum + product;
            end
        end
    end
    
    // Debug Log (Optional, keep matched with Golden if needed)
    /*
    always @(negedge weight_we) begin
         if (FAULT_TARGET == 0 && in_sum == 0 && product == 0) // Reducing spam
             $display("FAULTY PE Load Done: Time=%t | Loaded Val=%d", $time, weight_reg);
    end
    */

endmodule

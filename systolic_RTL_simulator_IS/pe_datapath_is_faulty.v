module pe_datapath_is_faulty #(
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

    reg [D_W-1:0] act_reg; // STATIONARY DATA (INPUT)
    wire [2*D_W-1:0] product;
    
    wire [D_W-1:0] effective_stream; // Corresponds to in_act (Now Streaming Weight from Left)
    wire [D_W-1:0] effective_stat;   // Corresponds to act_reg (Stationary Input)
    
    // Fault Logic
    // Target 0: Stream (Weight in IS context, flowing Horizontally)
    // Target 1: Stationary (Input in IS context, Stored)
    // Target 2: Psum
    
    generate
        if (FAULT_TARGET == 0) begin 
            // Fault on Stream (Weight - Left Port)
            assign effective_stream = in_act ^ fault_mask;
            assign effective_stat = act_reg;
        end else if (FAULT_TARGET == 1) begin
            // Fault on Stationary (Input)
            assign effective_stream = in_act;
            assign effective_stat = act_reg ^ fault_mask;
        end else begin
            assign effective_stream = in_act;
            assign effective_stat = act_reg;
        end
    endgenerate

    assign product = effective_stream * effective_stat;

    // PROBE LOGIC FOR Heuristic Script
    // PROBE LOGIC FOR Heuristic Script & Profiling
    always @(posedge clk) begin
         // Unconditional Log
         $display("Time %0d | PE[%0d][%0d] | In_Stat: %d | W_Stream: %d | Product: %d | Sum: %d", $time, ROW, COL, act_reg, in_act, product, out_sum);
    end

    always @(posedge clk) begin
        if (rst) begin
            act_reg <= 0;
            out_act <= 0;
            out_weight <= 0;
            out_sum <= 0;
        end else begin
            // Pass-throughs
            out_act <= in_act;       // Stream Weight Horizontal
            out_weight <= in_weight; // Pass Input Vertical (Fill)
            
            // Stream Pass-through (Faulty if Target 0 - Weight Stream)
            // Weight Stream is now on 'out_act' path logic?
            // Wait. 'effective_stream' is derived from 'in_act' (Left/Weight).
            // 'out_act' passes 'in_act'.
            if (FAULT_TARGET == 0) out_act <= effective_stream;
            else out_act <= in_act;

            // Load Logic (Reuse weight_we as load_input)
            if (weight_we) begin
                act_reg <= in_weight; // Load Input from Top
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

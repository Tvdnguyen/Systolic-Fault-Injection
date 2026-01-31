module pe_datapath_is #(
    parameter D_W = 8,
    parameter ROW = 0,
    parameter COL = 0
)(
    input wire clk,
    input wire rst,
    input wire weight_we, // Write Enable for Weight Reg
    
    input wire [D_W-1:0] in_act,
    input wire [D_W-1:0] in_weight, // From Top
    input wire [2*D_W-1:0] in_sum,
    
    output reg [D_W-1:0] out_act,
    output reg [D_W-1:0] out_weight,
    output reg [2*D_W-1:0] out_sum
);

    reg [D_W-1:0] act_reg;
    wire [2*D_W-1:0] product;
    
    // Computation: Stationary Input * Streaming Weight (Horizontal)
    // in_act is now the Weight Stream
    assign product = act_reg * in_act;

    `ifdef PROFILING
    always @(posedge clk) begin
        // Log Activity if in Compute Phase (!weight_we) and Operands are Non-Zero (Valid)
        // Note: We use Non-Zero data for Profiling. 0 implies Padding.
        if (!weight_we && act_reg != 0 && in_act != 0) begin
             $display("ACTIVE_LOG,%0d,%0d,%0d", ROW, COL, $time);
        end
    end
    `endif

    always @(posedge clk) begin
        if (rst) begin
            act_reg <= 0;
            out_act <= 0;
            out_weight <= 0;
            out_sum <= 0;
        end else begin
            // Pass-through Data
            out_act <= in_act;       // Stream Weight Horizontal
            out_weight <= in_weight; // Pass Input Vertical (during filling)
            
            // Loading Phase (Reuse weight_we as load_enable)
            if (weight_we) begin
                act_reg <= in_weight; // Load Input from Top (Vertical Shift)
                out_sum <= 0; 
            end else begin
                // Compute Phase
                out_sum <= in_sum + product;
            end
        end
    end
    
    always @(negedge weight_we) begin
         $display("GOLD PE Load Done: Time=%t | Loaded Act=%d", $time, act_reg);
    end

endmodule

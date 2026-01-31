module pe_datapath_ws #(
    parameter D_W = 8
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

    reg [D_W-1:0] weight_reg;
    wire [2*D_W-1:0] product;
    
    // Computation
    assign product = in_act * weight_reg;

    always @(posedge clk) begin
        if (rst) begin
            weight_reg <= 0;
            out_act <= 0;
            out_weight <= 0;
            out_sum <= 0;
        end else begin
            // Pass-through Data
            out_act <= in_act;
            out_weight <= in_weight;
            
            // Weight Loading
            if (weight_we) begin
                weight_reg <= in_weight;
                out_sum <= 0; // Reset sum during load
            end else begin
                // Compute
                out_sum <= in_sum + product;
            end
        end
    end
    
    always @(negedge weight_we) begin
         $display("GOLD PE Load Done: Time=%t | Loaded Val=%d", $time, weight_reg);
    end

endmodule

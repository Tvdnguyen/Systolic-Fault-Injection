module pe_is #(
    parameter D_W = 8,
    parameter ROW = 0,
    parameter COL = 0
)(
    input wire clk,
    input wire rst,
    input wire load_weight,       
    input wire [D_W-1:0] in_act,  
    input wire [D_W-1:0] in_weight, 
    input wire [2*D_W-1:0] in_sum, 
    
    output wire [D_W-1:0] out_act, 
    output wire [D_W-1:0] out_weight, 
    output wire [2*D_W-1:0] out_sum
);

    wire weight_we;

    // Bypass Control Instance to match Faulty Wrapper logic
    /*
    pe_control_is u_control (
        .clk(clk),
        .rst(rst),
        .load_weight(load_weight),
        .weight_we(weight_we)
    );
    */
    assign weight_we = load_weight;

    // Datapath Instance
    // Datapath Instance
    pe_datapath_is #(.D_W(D_W), .ROW(ROW), .COL(COL)) u_datapath (
        .clk(clk),
        .rst(rst),
        .weight_we(weight_we),
        .in_act(in_act),
        .in_weight(in_weight),
        .in_sum(in_sum),
        .out_act(out_act),
        .out_weight(out_weight),
        .out_sum(out_sum)
    );

endmodule

module pe_ws_faulty #(
    parameter D_W = 8,
    parameter FAULT_TARGET = 0, // 0:W, 1:I, 2:P
    parameter ROW = 0,
    parameter COL = 0
)(
    input wire clk,
    input wire rst,
    input wire weight_we, // Now: Input Load Enable
    input wire [D_W-1:0] in_act,    // Stationary Data Load Port
    input wire [D_W-1:0] in_weight, // Streaming Data Port
    input wire [2*D_W-1:0] in_sum,
    output wire [D_W-1:0] out_act,
    output wire [D_W-1:0] out_weight,
    output wire [2*D_W-1:0] out_sum,
    
    // Fault Interface
    input wire [D_W-1:0] fault_mask 
);

    pe_datapath_ws_faulty #(
        .D_W(D_W),
        .FAULT_TARGET(FAULT_TARGET),
        .ROW(ROW),
        .COL(COL)
    ) datapath_inst (
        .clk(clk), 
        .rst(rst), 
        .weight_we(weight_we), // Input Load
        .in_act(in_act),       // Stationary Input Load Value
        .in_weight(in_weight), // Streaming Weight
        .in_sum(in_sum), 
        .out_act(out_act), 
        .out_weight(out_weight), 
        .out_sum(out_sum),
        .fault_mask(fault_mask)
    );

    // Reuse control? 
    // pe_control_ws is just a passthrough in WS source?
    // Let's check pe_control_ws logic. It's empty/simple.
    // For now we assume control is inside datapath or simple.
    // pe_ws.v instantiated 'pe_control_ws'.
    // We can instantiate pe_control_ws here if needed, but Datapath has the main logic.
    
    pe_control_ws control_inst (
        .clk(clk),
        .rst(rst)
    );

endmodule

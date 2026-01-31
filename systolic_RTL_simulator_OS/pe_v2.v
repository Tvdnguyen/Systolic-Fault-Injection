// Module PE_V2 - Sử dụng Add-Shift Multiplier thay vì toán tử *
module pe_v2 #(
    parameter   D_W  = 8,
    parameter   I    = 1,
    parameter   J    = 1
) (
    input   wire                    clk,
    input   wire                    rst,
    input   wire                    init,
    input   wire     [D_W-1:0]       in_a,
    input   wire     [D_W-1:0]       in_b,
    output  wire     [D_W-1:0]       out_b,
    output  wire     [D_W-1:0]       out_a,

    input   wire     [2*D_W-1:0]   in_data,
    input   wire                    in_valid,
    output  wire     [2*D_W-1:0]   out_data,
    output  wire                    out_valid
);

    // --- Các dây kết nối (wires) giữa Control và Datapath ---
    wire init_r;
    wire in_valid_r;
    wire data_rsrv;
    wire out_stagevalid_wire;

    // --- Instantiate Datapath V2 (với Add-Shift Multiplier Combinational) ---
    pe_datapath_v2_comb_1 #(.D_W(D_W), .I(I), .J(J)) u_datapath (
        .clk                (clk),
        .rst                (rst),
        .in_a               (in_a),
        .in_b               (in_b),
        .in_data            (in_data),
        .in_valid           (in_valid),
        .init_r             (init_r),
        .in_valid_r         (in_valid_r),
        .data_rsrv          (data_rsrv),
        .out_a              (out_a),
        .out_b              (out_b),
        .out_data           (out_data),
        .out_stagevalid_out (out_stagevalid_wire)
    );

    // --- Instantiate Control Logic V2 (Refactored FSM) ---
    pe_control_v2 #(.D_W(D_W)) u_control (
        .clk                (clk),
        .rst                (rst),
        .init               (init),
        .in_valid           (in_valid),
        .out_stagevalid_from_dp(out_stagevalid_wire),
        .init_r             (init_r),
        .in_valid_r         (in_valid_r),
        .data_rsrv          (data_rsrv),
        .out_valid          (out_valid)
    );

endmodule

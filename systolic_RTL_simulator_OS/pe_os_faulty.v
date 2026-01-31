module pe_os_faulty #(
    parameter   D_W  = 8,
    parameter   I    = 1,
    parameter   J    = 1,
    parameter   FAULT_TARGET = 0, // 0:Weight, 1:Input, 2:Psum
    parameter   ROW = 0,
    parameter   COL = 0
) (
    input   wire                    clk,
    input   wire                    rst,
    input   wire                    init,
    input   wire     [D_W-1:0]       in_a,
    input   wire     [D_W-1:0]       in_b,
    input   wire     [D_W-1:0]       fault_mask, // Fault Injection Port
    output  wire     [D_W-1:0]       out_b,
    output  wire     [D_W-1:0]       out_a,
    input   wire     [2*D_W-1:0]   in_data,
    input   wire                    in_valid,
    output  wire     [2*D_W-1:0]   out_data,
    output  wire                    out_valid
);

    wire init_r;
    wire in_valid_r;
    wire data_rsrv;
    wire out_stagevalid_wire;

    // Instantiate FAULTY Datapath
    pe_datapath_os_faulty #(
        .D_W(D_W),
        .FAULT_TARGET(FAULT_TARGET),
        .ROW(ROW),
        .COL(COL)
    ) u_datapath (
        .clk                (clk),
        .rst                (rst),
        .in_a               (in_a),
        .in_b               (in_b),
        .in_data            (in_data),
        .in_valid           (in_valid),
        .fault_mask         (fault_mask), // Connect Fault Mask
        
        .init_r             (init_r),
        .in_valid_r         (in_valid_r),
        .data_rsrv          (data_rsrv),
        .out_a              (out_a),
        .out_b              (out_b),
        .out_data           (out_data),
        .out_stagevalid_out (out_stagevalid_wire)
    );

    // Reuse Standard Control V2
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

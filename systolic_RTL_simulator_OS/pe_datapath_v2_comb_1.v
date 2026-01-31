// Module Datapath V2 Combinational - Restructured Version
// Mỗi thanh ghi có always block riêng, sử dụng case statement
// MATCH latency với PE_V1 (1 cycle)
module pe_datapath_v2_comb_1 #(
    parameter D_W = 8,
    parameter I = 0,
    parameter J = 0
) (
    // Clock và Reset
    input  wire                clk,
    input  wire                rst,

    // Dữ liệu đầu vào từ bên ngoài
    input  wire [D_W-1:0]      in_a,
    input  wire [D_W-1:0]      in_b,
    input  wire [2*D_W-1:0]    in_data,
    input  wire                in_valid,

    // Tín hiệu điều khiển từ Control Logic
    input  wire                init_r,      // Đã đăng ký từ control
    input  wire                in_valid_r,  // Đã đăng ký từ control
    input  wire                data_rsrv,   // State từ control

    // Dữ liệu đầu ra
    output wire [D_W-1:0]      out_a,
    output wire [D_W-1:0]      out_b,
    output reg  [2*D_W-1:0]    out_data,
    output wire                out_stagevalid_out  // Feedback cho control
);

    // --- Internal Wires and Registers ---
    reg [D_W-1:0] a_tmp;
    reg [D_W-1:0] b_tmp;
    reg [2*D_W-1:0] in_data_r;
    reg [2*D_W-1:0] out_tmp;
    reg [2*D_W-1:0] out_tmp_r;
    reg [2*D_W-1:0] out_stage;
    reg out_stagevalid;

    // Multiplier signal - combinational
    wire [2*D_W-1:0] mult_product;

    // --- Add-Shift Multiplier Instantiation (Combinational) ---
    add_shift_multiplier_simple #(
        .D_W(D_W)
    ) u_multiplier (
        .a(in_a),
        .b(in_b),
        .product(mult_product)
    );

    // --- Register: a_tmp ---
    always @(posedge clk) begin
        if (rst)
            a_tmp <= 0;
        else
            a_tmp <= in_a;
    end

    // --- Register: b_tmp ---
    always @(posedge clk) begin
        if (rst)
            b_tmp <= 0;
        else
            b_tmp <= in_b;
    end

    // --- Register: in_data_r ---
    always @(posedge clk) begin
        if (rst)
            in_data_r <= 0;
        else
            in_data_r <= in_data;
    end

    // --- Register: out_tmp_r (multiplication result) ---
    always @(posedge clk) begin
        if (rst)
            out_tmp_r <= 0;
        else
            out_tmp_r <= mult_product;
    end

    // --- Register: out_tmp (accumulator) ---
    // Điều kiện: init_r
    always @(posedge clk) begin
        if (rst)
            out_tmp <= 0;
        else begin
            case (init_r)
                1'b1: out_tmp <= out_tmp_r;              // Initialize
                1'b0: out_tmp <= out_tmp + out_tmp_r;    // Accumulate
                default: out_tmp <= out_tmp + out_tmp_r;
            endcase
        end
    end

    // --- Register: out_stage ---
    // Điều kiện: {init_r, in_valid_r, data_rsrv}
    always @(posedge clk) begin
        if (rst)
            out_stage <= 0;
        else begin
            case ({init_r, in_valid_r, data_rsrv})
                3'b110: out_stage <= in_data_r;       // init_r=1, in_valid_r=1, data_rsrv=0
                3'b111: out_stage <= in_data_r;       // init_r=1, in_valid_r=1, data_rsrv=1
                3'b011: out_stage <= in_data_r;       // init_r=0, in_valid_r=1, data_rsrv=1
                3'b001: out_stage <= out_stage;       // init_r=0, in_valid_r=0, data_rsrv=1 - hold
                3'b100: out_stage <= out_stage;       // init_r=1, in_valid_r=0, data_rsrv=0 - hold
                3'b101: out_stage <= out_stage;       // init_r=1, in_valid_r=0, data_rsrv=1 - hold
                3'b010: out_stage <= out_stage;       // init_r=0, in_valid_r=1, data_rsrv=0 - hold
                3'b000: out_stage <= out_stage;       // init_r=0, in_valid_r=0, data_rsrv=0 - hold
                default: out_stage <= out_stage;
            endcase
        end
    end

    // --- Register: out_stagevalid ---
    // Điều kiện: {init_r, in_valid_r, data_rsrv}
    always @(posedge clk) begin
        if (rst)
            out_stagevalid <= 0;
        else begin
            case ({init_r, in_valid_r, data_rsrv})
                3'b110: out_stagevalid <= 1;    // init_r=1: Force Valid (Drain Trigger)
                3'b111: out_stagevalid <= 1;    
                3'b100: out_stagevalid <= 1;    
                3'b101: out_stagevalid <= 1;    
                3'b011: out_stagevalid <= in_valid_r;    // init_r=0, in_valid_r=1, data_rsrv=1
                default: out_stagevalid <= out_stagevalid;
            endcase
        end
    end

    // --- Register: out_data ---
    // Điều kiện: {init_r, in_valid_r, data_rsrv}
    always @(posedge clk) begin
        if (rst)
            out_data <= 0;
        else begin
            // Debug PE
            if (out_tmp > 0) $display("PE[%d][%d] Dump: Acc=%d OutData=%d Init=%b ValidIn=%b DataRsrv=%b", I, J, out_tmp, out_data, init_r, in_valid_r, data_rsrv);
            case ({init_r, in_valid_r, data_rsrv})
                3'b110: out_data <= out_tmp;        // init_r=1, in_valid_r=1, data_rsrv=0
                3'b111: out_data <= out_tmp;        // init_r=1, in_valid_r=1, data_rsrv=1
                3'b100: out_data <= out_tmp;        // init_r=1, in_valid_r=0, data_rsrv=0
                3'b101: out_data <= out_tmp;        // init_r=1, in_valid_r=0, data_rsrv=1
                3'b001: out_data <= out_stage;      // init_r=0, in_valid_r=0, data_rsrv=1
                3'b011: out_data <= out_stage;      // init_r=0, in_valid_r=1, data_rsrv=1
                3'b010: out_data <= in_data_r;      // init_r=0, in_valid_r=1, data_rsrv=0
                3'b000: out_data <= in_data_r;      // init_r=0, in_valid_r=0, data_rsrv=0
                default: out_data <= in_data_r;
            endcase
        end
    end

    // --- Output Assignments ---
    assign out_a = a_tmp;
    assign out_b = b_tmp;
    assign out_stagevalid_out = out_stagevalid;

endmodule

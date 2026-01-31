module systolic_ws_faulty #(
    parameter D_W = 8,
    parameter N = 4,
    parameter M = 4,
    parameter FAULT_ROW = 0,
    parameter FAULT_COL = 0,
    parameter FAULT_TARGET = 0 // 0:W, 1:I, 2:P
)(
    input wire clk,
    input wire rst,
    input wire load_weight, // Defines Input Loading Phase for IS
    input wire [D_W-1:0] m0 [N-1:0], // Connects to 'in_act' (Stationary Src? NO. Stream Src)
    input wire [D_W-1:0] m1 [N-1:0], // Connects to 'in_weight' (Vertical Load Src)
    
    // Fault Masks
    input wire [D_W-1:0] fault_masks [N-1:0][N-1:0],
    
    output wire [2*D_W-1:0] m2 [N-1:0]
);

    // Wire Matrix
    wire [D_W-1:0] horizontal_act [N:0][N:0];   // Passes Stream Act (Horizontal)
                                                 // Corresponds to m0
    wire [D_W-1:0] vertical_weight [N:0][N:0];   // Passes Loaded Stationary Input (Vertical)
                                                 // Corresponds to m1
    wire [2*D_W-1:0] vertical_sum [N:0][N:0];

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : row_logic
            // Left Boundary (Activations Input)
            assign horizontal_act[i][0] = m0[i];     // Feeds 'in_act' (Stream)
            
            for (j = 0; j < N; j = j + 1) begin : col_logic
                // Top Boundary
                if (i == 0) begin
                    assign vertical_weight[0][j] = m1[j];  // Feeds 'in_weight' (Vertical Load)
                    assign vertical_sum[0][j] = {2*D_W{1'b0}}; // Psum Init
                end
                
                // Uniform Instantiation: Use pe_ws_faulty everywhere for global logging
                pe_ws_faulty #(
                    .D_W(D_W),
                    .FAULT_TARGET(FAULT_TARGET),
                    .ROW(i),
                    .COL(j)
                ) pe (
                    .clk(clk),
                    .rst(rst),
                    .weight_we(load_weight), 
                    .in_act(horizontal_act[i][j]),       // Stream (Horiz)
                    .in_weight(vertical_weight[i][j]),   // Load (Vert)
                    .in_sum(vertical_sum[i][j]),
                    .out_act(horizontal_act[i][j+1]),
                    .out_weight(vertical_weight[i+1][j]),
                    .out_sum(vertical_sum[i+1][j]),
                    .fault_mask(fault_masks[i][j])
                );
                
                // Bottom Boundary Output Collection (Psum)
                if (i == N - 1) begin
                     assign m2[j] = vertical_sum[N][j]; 
                end
            end
        end
    endgenerate

endmodule

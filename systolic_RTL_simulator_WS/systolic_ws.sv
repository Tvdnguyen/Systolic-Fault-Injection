module systolic_ws #(
    parameter D_W = 8,
    parameter N = 4, // Array Size (NxN)
    parameter M = 4  // Memory/IO Width
)(
    input wire clk,
    input wire rst,
    input wire load_weight, // Global Weight Load Enable
    
    // Inputs
    input wire [D_W-1:0] m0 [N-1:0], // Left IO (Activations)
    input wire [D_W-1:0] m1 [N-1:0], // Top IO (Weights for Load / 0 for Compute)
    
    // Outputs
    output wire [2*D_W-1:0] m2 [N-1:0] // Bottom IO (Partial Sums)
);

    // Grid Wires
    wire [D_W-1:0] horizontal_act [N:0][N:0]; // [Row][Col]
    wire [D_W-1:0] vertical_weight [N:0][N:0]; // [Row][Col]
    wire [2*D_W-1:0] vertical_sum [N:0][N:0];   // [Row][Col]
    
    genvar r, c;
    generate
        for (r = 0; r < N; r = r + 1) begin : rows
            // Left Boundary (Activations Input)
            assign horizontal_act[r][0] = m0[r];
            
            for (c = 0; c < N; c = c + 1) begin : cols
                // Top Boundary
                if (r == 0) begin
                    assign vertical_weight[0][c] = m1[c]; // Weights In
                    assign vertical_sum[0][c] = {2*D_W{1'b0}}; // Psum Init (0)
                end
                
                // PE Instance
                pe_ws #(.D_W(D_W)) pe_inst (
                    .clk(clk),
                    .rst(rst),
                    .load_weight(load_weight),
                    
                    // Inputs
                    .in_act(horizontal_act[r][c]),
                    .in_weight(vertical_weight[r][c]),
                    .in_sum(vertical_sum[r][c]),
                    
                    // Outputs
                    .out_act(horizontal_act[r][c+1]),
                    .out_weight(vertical_weight[r+1][c]),
                    .out_sum(vertical_sum[r+1][c])
                );
                
                // Bottom Boundary Output Collection
                if (r == N - 1) begin
                    assign m2[c] = vertical_sum[N][c];
                end
            end
        end
    endgenerate

endmodule

// Simple Add-Shift Multiplier - Single cycle interface (combinational)
// WARNING: This is purely combinational - will have long critical path!
// For synthesis, use pipelined version instead.

module add_shift_multiplier_simple
#(
    parameter D_W = 8  // Data width
)
(
    input   wire  [D_W-1:0]     a,          // Multiplicand
    input   wire  [D_W-1:0]     b,          // Multiplier
    output  wire  [2*D_W-1:0]   product     // Result
);

    // Intermediate partial products
    wire [2*D_W-1:0] partial[D_W:0];
    wire [2*D_W-1:0] shifted_a[D_W-1:0];

    // Initial partial product is 0
    assign partial[0] = 0;

    // Generate shifted versions of a
    genvar i;
    generate
        for (i = 0; i < D_W; i = i + 1) begin : gen_shifts
            assign shifted_a[i] = {{D_W{1'b0}}, a} << i;

            // Add shifted_a if corresponding bit of b is 1
            assign partial[i+1] = partial[i] + (b[i] ? shifted_a[i] : 0);
        end
    endgenerate

    // Final product
    assign product = partial[D_W];

endmodule

module pe_control_is (
    input wire clk,
    input wire rst,
    input wire load_weight,
    output reg weight_we // Weight Write Enable
);

    // Simple Control Logic for now
    // If we add Valid/Ready protocol later, it goes here.
    always @(*) begin
        weight_we = load_weight;
    end

endmodule

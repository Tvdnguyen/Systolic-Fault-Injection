// Module Control Logic V2 - Explicit FSM Implementation
// Replaces complex if-else logic with a structured State Machine
module pe_control_v2 #(
    parameter D_W = 8
) (
    // Clock and Reset
    input  wire                clk,
    input  wire                rst,

    // Control Inputs
    input  wire                init,
    input  wire                in_valid,
    input  wire                out_stagevalid_from_dp,

    // Control Outputs
    output reg                 init_r,       // Registered init
    output reg                 in_valid_r,   // Registered in_valid
    output reg                 data_rsrv,    // State output (1 = DRAIN, 0 = NORMAL)
    output reg                 out_valid
);

    // --- State Encoding ---
    localparam S_NORMAL = 1'b0;
    localparam S_DRAIN  = 1'b1;

    reg state;
    reg next_state;

    // --- Input Registration ---
    always @(posedge clk) begin
        if (rst) begin
            init_r     <= 0;
            in_valid_r <= 0;
        end else begin
            init_r     <= init;
            in_valid_r <= in_valid;
        end
    end

    // --- State Register ---
    always @(posedge clk) begin
        if (rst) state <= S_NORMAL;
        else     state <= next_state;
    end

    // --- Next State Logic (Combinational) ---
    always @(*) begin
        next_state = state; // Default: hold state
        
        case (state)
            S_NORMAL: begin
                // Transition to DRAIN if init_r is high (Wavefront Trigger)
                // Relaxed condition: Do not wait for in_valid for First Column support.
                if (init_r) 
                    next_state = S_DRAIN;
            end
            
            S_DRAIN: begin
                 // Exit DRAIN? 
                 // If we drain for fixed cycles? Or until init drops?
                 // Init is a pulse (1 cycle).
                 // We need to stay in DRAIN for 1 cycle?
                 // Original logic: if (!in_valid_r) next_state = NORMAL.
                 // If Col 0, in_valid_r is ALWAYS 0.
                 // So it enters S_DRAIN, then NEXT cycle exiting S_DRAIN?
                 // This drains 1 value. Correct?
                 // Input stationary logic implies Psum accumulates. Output is 1 value.
                 // So 1 cycle drain is correct.
                 if (!init_r && !in_valid_r)
                     next_state = S_NORMAL;
            end
        endcase
    end

    // --- Output Logic (Registered) ---
    // Note: data_rsrv is effectively the state, but we register it explicitly 
    // to match the timing of the original module.
    always @(posedge clk) begin
        if (rst) begin
            data_rsrv <= 0;
            out_valid <= 0;
        end else begin
            // Default assignments
            data_rsrv <= (next_state == S_DRAIN); 
            
            case (state)
                S_NORMAL: begin
                    if (init_r) begin
                        // Transitioning to DRAIN (Master Trigger)
                        // If Input Valid is present (chained from prev), pass.
                        // If First Col (In Valid 0), we START valid.
                        // So force Valid High if init is High.
                        out_valid <= 1; // Force Valid (Aligns with init_r trigger) 
                    end else begin
                        // Pass-through valid signal
                        out_valid <= in_valid_r;
                    end
                end

                S_DRAIN: begin
                    // In DRAIN mode, output validity comes from datapath
                    out_valid <= out_stagevalid_from_dp;
                end
            endcase
        end
    end

endmodule

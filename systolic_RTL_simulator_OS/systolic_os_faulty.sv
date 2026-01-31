module systolic_os_faulty
#
(
    parameter   D_W  = 8,
    parameter   N   = 3,
    parameter   M   = 6,
    parameter   FAULT_ROW = 0,
    parameter   FAULT_COL = 0,
    parameter   FAULT_TARGET = 0 // 0:W, 1:I, 2:P
)
(
    input   wire                        clk,
    input   wire                        rst,
    input   wire                        enable_row_count_m0,
    output  wire    [$clog2(M)-1:0]     column_m0,
    output  wire    [$clog2(M/N)-1:0]   row_m0,
    output  wire    [$clog2(M/N)-1:0]   column_m1,
    output  wire    [$clog2(M)-1:0]     row_m1,
    input   wire    [D_W-1:0]           m0     [N-1:0],
    input   wire    [D_W-1:0]           m1     [N-1:0],
    
    // Fault Masks 
    input   wire    [D_W-1:0]           fault_masks [N-1:0][N-1:0],
    
    output  wire    [2*D_W-1:0]         m2     [N-1:0]  ,
    output   wire    [N-1:0]            valid_m2
);


counter#
(
    .WIDTH  (M),
    .HEIGHT (1)
)
counter_m1
(

    .clk                      (clk),
    .rst                      (rst),
    .enable_row_count         (1'b1),
    .column_counter           (row_m1),
    .row_counter              (column_m1)
);


counter#
(
    .WIDTH  (M),
    .HEIGHT (1)
)
counter_m0
(

    .clk                (clk),
    .rst                (rst),
    .enable_row_count   (enable_row_count_m0),
    .column_counter     (column_m0),
    .row_counter        (row_m0)
);

integer slice;
integer i;
integer j;
integer z;
reg init[N-1:0][N-1:0];



	// Reverse Column Draining Mechanism
	// Triggers 'init' for column N-1, then N-2... down to 0.
	// This ensures clean shifting of data to the right without collision.
	always@(posedge clk) begin
		if(rst) begin
			slice <= N; // Start outside valid range (N-1 is first valid)
			for(i=0; i<N; i=i+1)
				for(j=0; j<N; j=j+1)
					init[i][j] <= 0;
		end
		else begin
			// Reset init pulse
			for(i=0; i<N; i=i+1)
				for(j=0; j<N; j=j+1)
					init[i][j] <= 0;

			// Trigger Condition: 
			// Wait for computation to finish (column_m0 reaches M-1).
			// Then start draining sequence.
			if(column_m0 == M-1)
				slice <= N-1; // Set to last column

			// Draining Sequence
			if(slice >= 0 && slice < N) begin
				for(i=0; i<N; i=i+1) begin
					// Activate entire column 'slice'
					init[i][slice] <= 1; 
				end
				slice <= slice - 1; // Move left
			end 
		end 
	end



genvar k,l;
wire [D_W-1:0] HorizontalWire[N:0][N:0];
wire [D_W-1:0] VerticalWire[N:0][N:0];
wire [2*D_W-1:0] OutputDataWire[N:0][N:0];
wire  OutputValidWire[N:0][N:0];

    generate
    for(k=0;k<N;k=k+1) begin:row
        assign HorizontalWire[k][0]=m0[k];
        assign VerticalWire[0][k]=m1[k];
        assign OutputDataWire[k][0]=0;
        assign OutputValidWire[k][0]=0;
        assign m2[k]=OutputDataWire[k][N];
        assign valid_m2[k]=OutputValidWire[k][N];
        
        for(l=0;l<N;l=l+1)begin:col
            // Uniform Instantiation for Profiling
            pe_os_faulty #(
                .D_W(D_W), .FAULT_TARGET(FAULT_TARGET),
                .ROW(k), .COL(l)
            ) pe_inst (
                .clk(clk),
                .rst(rst),
                // Wiring
                .in_a(HorizontalWire[k][l]),
                .in_b(VerticalWire[k][l]),
                .out_a(HorizontalWire[k][l+1]),
                .out_b(VerticalWire[k+1][l]),
                .in_data(OutputDataWire[k][l]),
                .in_valid(OutputValidWire[k][l]),
                .out_valid(OutputValidWire[k][l+1]),
                .out_data(OutputDataWire[k][l+1]),
                .init(init[k][l]),
                // Fault Injection
                .fault_mask(fault_masks[k][l])
            );
        end //end for col
    end //end for row
endgenerate


endmodule

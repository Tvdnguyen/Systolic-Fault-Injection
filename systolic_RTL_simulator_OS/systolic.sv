module systolic
#
(
    parameter   D_W  = 8,
    parameter   N   = 3,
    parameter   M   = 6
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



	// Reverse Column Draining Mechanism (synced with systolic_os_faulty.sv)
	always@(posedge clk) begin
        // Internal Debug Removed
		if(rst) begin
			slice <= N; // Start outside valid range
			for(i=0; i<N; i=i+1)
				for(j=0; j<N; j=j+1)
					init[i][j] <= 0;
		end
		else begin
			// Reset init pulse
			for(i=0; i<N; i=i+1)
				for(j=0; j<N; j=j+1)
					init[i][j] <= 0;

			// Trigger Condition
			if(column_m0 == M-1) begin
				slice <= N-1; // Set to last column
            end
			// Draining Sequence
			else if(slice >= 0 && slice < N) begin
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
pe_v2 #(.D_W(D_W), .I(k), .J(l)) pe_inst(.clk(clk),
	.rst(rst),
	.in_a(HorizontalWire[k][l]),
	.in_b(VerticalWire[k][l]),
	.out_a(HorizontalWire[k][l+1]),
	.out_b(VerticalWire[k+1][l]),
	.in_data(OutputDataWire[k][l]),
	.in_valid(OutputValidWire[k][l]),
	.out_valid(OutputValidWire[k][l+1]),
	.out_data(OutputDataWire[k][l+1]),
	.init(init[k][l]));

end //end for l loop



end //end for k loop

endgenerate


endmodule

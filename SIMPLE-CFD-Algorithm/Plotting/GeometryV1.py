import matplotlib.pyplot as plt

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
outfile = "geometry.txt"

# Grid Dimensions (MUST MATCH SIMPLE.h)
M = 300   # Height (y)
N = 400   # Width (x)

# Buffer Settings
N_inlet_buffer = 100
N_outlet_buffer = 0  # Increased buffer to 200 as requested

# Channel Settings
num_channels = 20
channel_width = 10

# Calculate y-positions for the 4 channels to be EQUALLY SPACED
# Spacing divides the height M into (num_channels + 1) equal segments
spacing = M // (num_channels + 1)
channel_y_centers = [spacing * (k + 1) for k in range(num_channels)]

# -----------------------------------------------------------
# GEOMETRY GENERATION
# -----------------------------------------------------------
grid_data = [] 
text_lines = []

print(f"Generating {M}x{N} Network Geometry...")
print(f"Channels: {num_channels}, Spacing: {spacing}")
print(f"Outlet Buffer: {N_outlet_buffer}")

for i in range(M):
    row_vals = []
    row_ints = []
    
    for j in range(N):
        
        # 1. Default Background = SOLID (0) (These act as the walls)
        val = 0 
        
        # ---------------------------------------------------
        # 2. Draw the Horizontal Channels (Fluid = 1)
        # ---------------------------------------------------
        for y_c in channel_y_centers:
            # If i is within the channel width centered at y_c
            if y_c - channel_width//2 <= i <= y_c + channel_width//2:
                val = 1

        # ---------------------------------------------------
        # 3. Overwrite Inlet and Outlet Buffers (Force Fluid)
        # ---------------------------------------------------
        # Inlet Plenum (Left)
        if j < N_inlet_buffer:
            val = 1
            
        # Outlet Plenum (Right) - Now 200 cells wide
        if j >= (N - N_outlet_buffer):
            val = 1

        # ---------------------------------------------------
        # 4. Force Top/Bottom Domain Boundary Walls (Safety)
        # ---------------------------------------------------
        if i == 0 or i == M - 1:
            val = 0

        row_vals.append(str(val))
        row_ints.append(val)

    grid_data.append(row_ints)
    text_lines.append(" ".join(row_vals))

# -----------------------------------------------------------
# WRITE TO FILE
# -----------------------------------------------------------
with open(outfile, "w") as f:
    for line in text_lines:
        f.write(line + "\n")

print(f"Successfully wrote {outfile}")

# -----------------------------------------------------------
# VISUALIZATION (Optional: Check the result)
# -----------------------------------------------------------
try:
    plt.imshow(grid_data, cmap='gray', origin='lower')
    plt.title(f"Geometry {M}x{N} (Buffer={N_outlet_buffer})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
except:
    print("Matplotlib not available or display failed. File written successfully.")
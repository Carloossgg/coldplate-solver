import matplotlib.pyplot as plt

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
outfile = "geometry.txt"

# Grid Dimensions (MUST MATCH SIMPLE.h)
M = 100   # Height
N = 400   # Width (Increased to 400 to fit the new buffer)

# Buffer Settings
N_inlet_buffer = 10
N_outlet_buffer = 100  # The 100 cells at the end you requested

# Channel Settings
num_channels = 4
channel_width = 10

# Calculate y-positions for the 4 channels
spacing = M // (num_channels + 1)
channel_y_centers = [spacing * (k + 1) for k in range(num_channels)]

# -----------------------------------------------------------
# GEOMETRY GENERATION
# -----------------------------------------------------------
grid_data = [] 
text_lines = []

print(f"Generating {M}x{N} Network Geometry...")

for i in range(M):
    row_vals = []
    row_ints = []
    
    for j in range(N):
        
        # 1. Default Background = SOLID (0)
        val = 0 
        
        # ---------------------------------------------------
        # 2. Draw the 4 Horizontal Channels (Fluid)
        # ---------------------------------------------------
        for y_c in channel_y_centers:
            if y_c - channel_width//2 <= i <= y_c + channel_width//2:
                val = 1

        # ---------------------------------------------------
        # 3. Add Vertical Interconnects (The "Meeting Points")
        # ---------------------------------------------------
        # Only draw these in the "Body" region (not in buffers)
        if j > N_inlet_buffer and j < (N - N_outlet_buffer):
            
            # Junction A (x=60)
            if 55 <= j <= 65:
                y0, y1 = channel_y_centers[0], channel_y_centers[1]
                if y0 <= i <= y1: val = 1

            # Junction B (x=120)
            if 115 <= j <= 125:
                y2, y3 = channel_y_centers[2], channel_y_centers[3]
                if y2 <= i <= y3: val = 1

            # Junction C (x=180)
            if 175 <= j <= 185:
                y1, y2 = channel_y_centers[1], channel_y_centers[2]
                if y1 <= i <= y2: val = 1

            # Junction D (x=240)
            if 235 <= j <= 245:
                y_top, y_bot = channel_y_centers[0], channel_y_centers[3]
                if y_top <= i <= y_bot: val = 1

        # ---------------------------------------------------
        # 4. Create "Forced Detours" (Blockages)
        # ---------------------------------------------------
        # BLOCKAGE 1 (x=80, Channel 1)
        if 80 <= j <= 90:
            y_c1 = channel_y_centers[1]
            if y_c1 - channel_width//2 <= i <= y_c1 + channel_width//2:
                val = 0 

        # BLOCKAGE 2 (x=200, Channel 3)
        if 200 <= j <= 210:
            y_c3 = channel_y_centers[3]
            if y_c3 - channel_width//2 <= i <= y_c3 + channel_width//2:
                val = 0

        # ---------------------------------------------------
        # 5. Overwrite Inlet and Outlet Buffers (Force Fluid)
        # ---------------------------------------------------
        # Inlet Plenum
        if j < N_inlet_buffer:
            val = 1
            
        # Outlet Plenum (The 100 buffer at the end)
        if j >= (N - N_outlet_buffer):
            val = 1

        # Force Top/Bottom Domain Walls (Safety)
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

print(f"Successfully wrote {outfile} with size {M}x{N}")

# -----------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.title(f"Network Geometry ({M}x{N})\nWhite = Fluid, Black = Solid")
plt.imshow(grid_data, cmap='gray', origin='upper', aspect='equal')
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
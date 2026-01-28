import numpy as np
import os
import gmsh

def export_to_step(input_dir="ExportFiles", output_file="geometry.step"):
    """
    Reads geometry_fluid.txt and fluid_params.txt, optimizes the solid 
    regions into rectangles, and exports a TRUE 3D STEP file using gmsh 
    (OpenCASCADE kernel).
    """
    geom_path = os.path.join(input_dir, "geometry_fluid.txt")
    params_path = os.path.join(input_dir, "fluid_params.txt")

    if not os.path.exists(geom_path) or not os.path.exists(params_path):
        print(f"Error: Required files not found in {input_dir}")
        return

    # 1. Load Data
    print(f"Loading geometry from {geom_path}...")
    geometry = np.loadtxt(geom_path, dtype=int)
    
    with open(params_path, 'r') as f:
        params = f.readline().split()
        M = int(params[0])
        N = int(params[1])
        dy = float(params[2])
        dx = float(params[3])
        Ht = float(params[7]) if len(params) >= 8 else 0.004

    # --- SCALE TO MILLIMETERS ---
    # CAD software (especially CATIA) handles MM much better than Meters 
    # for small-scale parts. Meters lead to precision/tolerance errors.
    scale = 1000.0 
    dx_mm = dx * scale
    dy_mm = dy * scale
    Ht_mm = Ht * scale
    base_plate_thickness_mm = 0.0005 * scale # 0.5 mm

    print(f"Grid: {N}x{M}, dx={dx_mm:.4f}mm, dy={dy_mm:.4f}mm, Ht={Ht_mm:.4f}mm")

    # 2. Greedy Optimization: Merge adjacent solid cells into rectangles
    solids = (geometry == 1)
    visited = np.zeros_like(solids, dtype=bool)
    rects = []

    print("Optimizing geometry into rectangles...")
    for j in range(M):
        for i in range(N):
            if solids[j, i] and not visited[j, i]:
                w = 1
                while i + w < N and solids[j, i + w] and not visited[j, i + w]:
                    w += 1
                
                h = 1
                while j + h < M:
                    if np.all(solids[j + h, i : i + w]) and not np.any(visited[j + h, i : i + w]):
                        h += 1
                    else:
                        break
                
                visited[j : j + h, i : i + w] = True
                rects.append((i, j, w, h))

    print(f"Reduced {np.sum(solids)} cells to {len(rects)} rectangular blocks.")

    # 3. Create 3D Model with Gmsh (OCC Kernel)
    print("Generating 3D STEP model via gmsh (UNITS: MM)...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("CFD_Geometry")

    tags = []
    
    # --- ADD BASE PLATE ---
    print(f"Adding {base_plate_thickness_mm}mm base plate...")
    tag_base = gmsh.model.occ.addBox(0, 0, -base_plate_thickness_mm, N*dx_mm, M*dy_mm, base_plate_thickness_mm)
    tags.append((3, tag_base))

    total = len(rects)
    for idx, (i, j, w, h) in enumerate(rects):
        x_phys = i * dx_mm
        y_phys = j * dy_mm
        w_phys = w * dx_mm
        h_phys = h * dy_mm
        
        # Add 3D box to the OpenCASCADE kernel
        tag = gmsh.model.occ.addBox(x_phys, y_phys, 0, w_phys, h_phys, Ht_mm)
        tags.append((3, tag))
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total} rectangles...")

    print("Synchronizing OCC model...")
    gmsh.model.occ.synchronize()

    # --- CLEANUP AND FUSE ---
    print("Removing duplicates and healing geometry...")
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    if len(tags) > 1:
        print(f"Fusing {len(tags)} blocks into a single solid body...")
        try:
            # Boolean Union
            gmsh.model.occ.fuse([tags[0]], tags[1:])
            gmsh.model.occ.synchronize()
            print("Fusion complete.")
        except Exception as e:
            print(f"Warning: Fusion failed ({e}). Exporting separate blocks.")
    
    # Ensure we have volumes (solids)
    vols = gmsh.model.getEntities(dim=3)
    print(f"Final model has {len(vols)} solid volume(s).")

    # 4. Save to STEP file
    print(f"Saving to {output_file} (Dimensions are in MM)...")
    gmsh.write(output_file)
    
    gmsh.finalize()
    print("Export complete!")

if __name__ == "__main__":
    export_to_step(output_file="ExportFiles/geometry_3d.step")


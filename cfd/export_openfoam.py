#!/usr/bin/env python3
"""
Export optimized geometry to OpenFOAM format for high-fidelity CFD simulation
"""

import numpy as np
import os
from typing import Tuple

def export_stl_geometry(xPhys: np.ndarray, 
                       Lx: float, Ly: float, H: float,
                       threshold: float = 0.5,
                       output_file: str = "coldplate.stl"):
    """
    Export the optimized design as STL file for OpenFOAM meshing
    
    Args:
        xPhys: Optimized design field (nely x nelx)
        Lx, Ly: Physical dimensions (m)
        H: Channel height (m)
        threshold: Density threshold to define fluid/solid boundary
        output_file: Output STL filename
    """
    nely, nelx = xPhys.shape
    dx = Lx / nelx
    dy = Ly / nely
    
    # Extract fluid region (marching squares algorithm)
    fluid_cells = xPhys > threshold
    
    vertices = []
    faces = []
    
    # Generate STL for the fluid channels (3D extrusion)
    # This creates a 3D geometry by extruding the 2D design
    
    print(f"Generating 3D geometry from 2D design...")
    print(f"Domain: {Lx*1000:.1f} x {Ly*1000:.1f} x {H*1000:.1f} mm")
    print(f"Threshold: {threshold}")
    
    # Create vertices and faces for bottom and top surfaces
    vertex_count = 0
    
    for i in range(nely):
        for j in range(nelx):
            if fluid_cells[i, j]:
                # Define 8 vertices of the hexahedron (voxel)
                x0, y0 = j * dx, i * dy
                x1, y1 = (j + 1) * dx, (i + 1) * dy
                z0, z1 = 0, H
                
                # 8 vertices of the voxel
                verts = [
                    [x0, y0, z0],  # 0: bottom-left-front
                    [x1, y0, z0],  # 1: bottom-right-front
                    [x1, y1, z0],  # 2: bottom-right-back
                    [x0, y1, z0],  # 3: bottom-left-back
                    [x0, y0, z1],  # 4: top-left-front
                    [x1, y0, z1],  # 5: top-right-front
                    [x1, y1, z1],  # 6: top-right-back
                    [x0, y1, z1],  # 7: top-left-back
                ]
                
                # Check neighbors to determine which faces to create
                # Only create faces on boundaries (interfaces with solid)
                
                # Bottom face (z = 0)
                if i == 0 or not fluid_cells[i-1, j]:
                    # This is always a boundary in z
                    pass  # Bottom is inlet/outlet
                
                # Top face (z = H)
                pass  # Top is always solid boundary
                
                # Front face (y = y0)
                if j == 0 or not fluid_cells[i, j-1]:
                    base_idx = len(vertices)
                    vertices.extend([verts[0], verts[1], verts[5], verts[4]])
                    faces.append([base_idx, base_idx+1, base_idx+2])
                    faces.append([base_idx, base_idx+2, base_idx+3])
                
                # Back face (y = y1)
                if j == nelx-1 or not fluid_cells[i, j+1]:
                    base_idx = len(vertices)
                    vertices.extend([verts[3], verts[2], verts[6], verts[7]])
                    faces.append([base_idx, base_idx+1, base_idx+2])
                    faces.append([base_idx, base_idx+2, base_idx+3])
                
                # Left face (x = x0)
                if i == 0 or not fluid_cells[i-1, j]:
                    base_idx = len(vertices)
                    vertices.extend([verts[0], verts[4], verts[7], verts[3]])
                    faces.append([base_idx, base_idx+1, base_idx+2])
                    faces.append([base_idx, base_idx+2, base_idx+3])
                
                # Right face (x = x1)
                if i == nely-1 or not fluid_cells[i+1, j]:
                    base_idx = len(vertices)
                    vertices.extend([verts[1], verts[2], verts[6], verts[5]])
                    faces.append([base_idx, base_idx+1, base_idx+2])
                    faces.append([base_idx, base_idx+2, base_idx+3])
    
    # Write STL file (ASCII format)
    with open(output_file, 'w') as f:
        f.write(f"solid coldplate\n")
        
        for face in faces:
            # Get vertices for this face (triangle)
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Calculate normal vector
            edge1 = np.array(v1) - np.array(v0)
            edge2 = np.array(v2) - np.array(v0)
            normal = np.cross(edge1, edge2)
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 0:
                normal = normal / norm_mag
            else:
                normal = [0, 0, 1]
            
            # Write facet
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write(f"    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write(f"    endloop\n")
            f.write(f"  endfacet\n")
        
        f.write(f"endsolid coldplate\n")
    
    print(f"\nSTL file exported: {output_file}")
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    

def create_openfoam_case(case_dir: str, 
                        xPhys: np.ndarray,
                        params: dict,
                        props: dict):
    """
    Create a complete OpenFOAM case directory for CFD simulation
    
    Args:
        case_dir: Directory to create OpenFOAM case
        xPhys: Optimized design
        params: Optimization parameters
        props: Physical properties
    """
    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(f"{case_dir}/0", exist_ok=True)
    os.makedirs(f"{case_dir}/constant", exist_ok=True)
    os.makedirs(f"{case_dir}/system", exist_ok=True)
    
    print(f"\nCreating OpenFOAM case in {case_dir}/")
    
    # Export geometry
    export_stl_geometry(xPhys, params['Lx'], params['Ly'], props['H'],
                       output_file=f"{case_dir}/constant/geometry.stl")
    
    # Create 0/U (velocity initial condition)
    with open(f"{case_dir}/0/U", 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    inlet
    {
        type            pressureInletVelocity;
        value           uniform (0 0 0);
    }

    outlet
    {
        type            zeroGradient;
    }

    walls
    {
        type            noSlip;
    }
}
""")
    
    # Create 0/p (pressure initial condition)
    with open(f"{case_dir}/0/p", 'w') as f:
        p_inlet = params.get('p_inlet', 1000.0)
        f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {p_inlet};
    }}

    outlet
    {{
        type            fixedValue;
        value           uniform 0;
    }}

    walls
    {{
        type            zeroGradient;
    }}
}}
""")
    
    # Create 0/T (temperature initial condition)
    with open(f"{case_dir}/0/T", 'w') as f:
        T_inlet = params.get('T_inlet', 20.0) + 273.15  # Convert to Kelvin
        f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 1 0 0 0];

internalField   uniform {T_inlet};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {T_inlet};
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    walls
    {{
        type            fixedGradient;
        gradient        uniform 0;  // Will be set from power map
    }}
}}
""")
    
    # Create constant/transportProperties
    with open(f"{case_dir}/constant/transportProperties", 'w') as f:
        rho = props.get('rho_f', 998.0)
        mu = props.get('mu_f', 0.001)
        cp = props.get('cp_f', 4184.0)
        k = props.get('k_f', 0.598)
        
        nu = mu / rho  # kinematic viscosity
        alpha = k / (rho * cp)  # thermal diffusivity
        
        f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              [{nu:.6e}];
rho             [{rho:.6e}];
Cp              [{cp:.6e}];
kappa           [{k:.6e}];
alpha           [{alpha:.6e}];
""")
    
    # Create system/controlDict
    with open(f"{case_dir}/system/controlDict", 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     buoyantSimpleFoam;

startFrom       latestTime;

startTime       0;

stopAt          endTime;

endTime         2000;

deltaT          1;

writeControl    timeStep;

writeInterval   100;

purgeWrite      2;

writeFormat     ascii;

writePrecision  8;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;
""")
    
    # Create system/fvSchemes
    with open(f"{case_dir}/system/fvSchemes", 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,T)      bounded Gauss linearUpwind grad(T);
    div(phi,K)      bounded Gauss limitedLinear 1;
    div(phi,h)      bounded Gauss limitedLinear 1;
    div(phi,k)      bounded Gauss limitedLinear 1;
    div(phi,epsilon) bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
""")
    
    # Create system/fvSolution
    with open(f"{case_dir}/system/fvSolution", 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-6;
        relTol          0.01;
        smoother        GaussSeidel;
    }

    "(U|T|k|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        T               1e-4;
    }
}

relaxationFactors
{
    equations
    {
        U               0.7;
        T               0.7;
    }
}
""")
    
    # Create README
    with open(f"{case_dir}/README.md", 'w') as f:
        f.write(f"""# OpenFOAM Cold Plate Simulation Case

## Case Description
This case simulates fluid flow and heat transfer in the optimized microfluidic cold plate.

- Domain: {params['Lx']*1000:.1f} x {params['Ly']*1000:.1f} x {props['H']*1000:.1f} mm
- Inlet pressure: {params.get('p_inlet', 1000):.0f} Pa
- Inlet temperature: {params.get('T_inlet', 20):.1f} °C
- Fluid: Water

## Running the Simulation

### 1. Mesh Generation
```bash
surfaceFeatureExtract
blockMesh
snappyHexMesh -overwrite
```

### 2. Run Simulation
```bash
buoyantSimpleFoam
```

### 3. Post-processing
```bash
paraFoam
```

## Notes
- The STL geometry is located in constant/geometry.stl
- Boundary conditions are defined in the 0/ directory
- Solver settings in system/controlDict, fvSchemes, fvSolution
""")
    
    print(f"\nOpenFOAM case created successfully!")
    print(f"Location: {case_dir}/")
    print("\nNext steps:")
    print("  1. cd {case_dir}")
    print("  2. Run meshing: blockMesh && snappyHexMesh")
    print("  3. Run solver: buoyantSimpleFoam")
    print("  4. Visualize: paraFoam")


if __name__ == "__main__":
    # Load optimized design
    import sys
    
    results_dir = '/home/claude'
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    
    if os.path.exists(f'{results_dir}/optimized_design.npy'):
        print("Loading optimization results...")
        
        xPhys = np.load(f'{results_dir}/optimized_design.npy')
        
        # Example parameters (would normally come from saved config)
        params = {
            'Lx': 0.10,
            'Ly': 0.10,
            'p_inlet': 1000.0,
            'T_inlet': 20.0,
        }
        
        props = {
            'H': 0.005,
            'rho_f': 998.0,
            'mu_f': 0.001,
            'cp_f': 4184.0,
            'k_f': 0.598,
        }
        
        # Create OpenFOAM case
        case_dir = f"{results_dir}/openfoam_case"
        create_openfoam_case(case_dir, xPhys, params, props)
        
    else:
        print(f"No results found in {results_dir}/")
        print("Please run topology_optimizer.py first.")
// ============================================================================
// File: Utilities/output.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Handles all file I/O and data export for the CFD solver, including:
//   - Per-iteration console output and log file writing
//   - Final solution field export (u, v, p as text and VTK)
//   - Thermal domain cropping for coupled thermal analysis
//
// OUTPUT FILES:
//   1. ExportFiles/residuals.txt: Convergence history log
//      Format: Iter Mass U V Core_dP Full_dP
//
//   2. ExportFiles/pressure_drop_history.txt: Pressure drop evolution
//      Format: Iter Core_Total Full_Total Core_Static Full_Static
//
//   3. ExportFiles/u.txt, v.txt, p.txt: Raw staggered fields (for restart)
//
//   4. ExportFiles/u_full.txt, v_full.txt: Cell-centered velocity fields
//
//   5. ExportFiles/u_thermal.txt, v_thermal.txt, pressure_thermal.txt:
//      Cropped to thermal domain (heatsink region only, excluding buffers)
//
//   6. ExportFiles/fluid_results.vtk: Visualization file for ParaView
//      Contains: pressure, velocity vectors, cellType, gamma (density), alpha
//
// THERMAL CROPPING:
//   The CFD domain may include inlet/outlet buffer zones for flow development.
//   For thermal analysis, we crop to just the heatsink region:
//   - Skip N_in_buffer columns from left (inlet buffer)
//   - Skip N_out_buffer columns from right (outlet buffer)
//   - Export only the central N_thermal = N - N_in_buffer - N_out_buffer columns
//
// VTK FORMAT:
//   Uses Legacy VTK ASCII format (STRUCTURED_POINTS) for compatibility.
//   Cell data includes scalar fields and velocity vectors.
//
// ============================================================================
#include "SIMPLE.h"
#include <iomanip>
#include <algorithm>
#include <cmath>

// ============================================================================
// saveMatrix: Write a 2D Eigen matrix to a text file
// ============================================================================
void SIMPLE::saveMatrix(Eigen::MatrixXf inputMatrix, std::string fileName)
{
    std::string fullPath = "ExportFiles/" + fileName + ".txt";
    std::ofstream out(fullPath);
    if (!out.is_open()) {
        std::cerr << "Error: could not open file '" << fullPath << "' for writing.\n";
        return;
    }

    int rows = static_cast<int>(inputMatrix.rows());
    int cols = static_cast<int>(inputMatrix.cols());

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out << inputMatrix(i, j);
            if (j < cols - 1) {
                out << "\t";
            }
        }
        out << "\n";
    }
}

// ------------------------------------------------------------------
// Main Save Function (Includes Thermal Slicing & VTK)
// ------------------------------------------------------------------
void SIMPLE::saveAll()
{
    ScopedTimer t("Output: saveAll (Total)");
    const int physRows = M - 1;
    const int physCols = N - 1;
    
    // Optionally save raw staggered fields for restart
    {
        ScopedTimer t2("Output: Staggered Fields (txt)");
        saveMatrix(u, "u");
        saveMatrix(v, "v");
        saveMatrix(p, "p");
    }

    // 1. Build cell-aligned fields for export (Required for VTK & Thermal)
    // For topology optimization: compute values everywhere.
    Eigen::MatrixXf uCenter = Eigen::MatrixXf::Zero(physRows, physCols);
    Eigen::MatrixXf vCenter = Eigen::MatrixXf::Zero(physRows, physCols);
    Eigen::MatrixXf pCenter = Eigen::MatrixXf::Zero(physRows, physCols);
    Eigen::MatrixXf uResidCenter = Eigen::MatrixXf::Zero(physRows, physCols);
    Eigen::MatrixXf vResidCenter = Eigen::MatrixXf::Zero(physRows, physCols);

    {
        ScopedTimer t2("Output: Cell-Aligned Field Assembly");
        for (int i = 0; i < physRows; ++i) {
            for (int j = 0; j < physCols; ++j) {
                // Convert staggered velocities to cell centers using the correct
                // face pair directions:
                // - u is on vertical faces -> average WEST/EAST faces (vary j)
                // - v is on horizontal faces -> average SOUTH/NORTH faces (vary i)
                const int uRow = i + 1;   // interior u row aligned with physical cell row i
                const int vCol = j + 1;   // interior v col aligned with physical cell col j
                uCenter(i, j) = 0.5f * (u(uRow, j) + u(uRow, j + 1));
                vCenter(i, j) = 0.5f * (v(i, vCol) + v(i + 1, vCol));

                // Momentum residuals: per-cell |u - u_old| and |v - v_old| averaged to centers
                // using the same face-pair averaging as the velocity export.
                const float duW = u(uRow, j)     - uOld(uRow, j);
                const float duE = u(uRow, j + 1) - uOld(uRow, j + 1);
                const float dvS = v(i, vCol)     - vOld(i, vCol);
                const float dvN = v(i + 1, vCol) - vOld(i + 1, vCol);
                uResidCenter(i, j) = 0.5f * (std::abs(duW) + std::abs(duE));
                vResidCenter(i, j) = 0.5f * (std::abs(dvS) + std::abs(dvN));

                // Pressure at physical cell center (offset by 1 due to ghost padding).
                pCenter(i, j) = p(i + 1, j + 1);
            }
        }
    }

    // 2. Save Full Domain Data (Text format)
    {
        ScopedTimer t2("Output: Full Domain Fields (txt)");
        saveMatrix(uCenter, "u_full");
        saveMatrix(vCenter, "v_full");
        saveMatrix(pCenter, "pressure_full");
        saveMatrix(uResidCenter, "u_resid_full");
        saveMatrix(vResidCenter, "v_resid_full");
    }
    
    // ---------------------------------------------------------
    // 3. CALCULATE PRESSURE GRADIENT (FULL DOMAIN)
    // ---------------------------------------------------------
    Eigen::MatrixXf pGradX = Eigen::MatrixXf::Zero(physRows, physCols);  // dp/dx
    Eigen::MatrixXf pGradY = Eigen::MatrixXf::Zero(physRows, physCols);  // dp/dy
    Eigen::MatrixXf pGradMag = Eigen::MatrixXf::Zero(physRows, physCols); // |∇p|
    
    {
        ScopedTimer t2("Output: Pressure Gradient Calculation");
        for (int i = 0; i < physRows; ++i) {
            for (int j = 0; j < physCols; ++j) {
                // Compute gradient for ALL cells (topology optimization consistency)
                    // dp/dx using central difference where possible
                    if (j > 0 && j < physCols - 1) {
                        pGradX(i, j) = (pCenter(i, j + 1) - pCenter(i, j - 1)) / (2.0f * hx);
                    } else if (j == 0) {
                        pGradX(i, j) = (pCenter(i, j + 1) - pCenter(i, j)) / hx;  // Forward diff
                    } else {
                        pGradX(i, j) = (pCenter(i, j) - pCenter(i, j - 1)) / hx;  // Backward diff
                    }
                    
                    // dp/dy using central difference where possible
                    if (i > 0 && i < physRows - 1) {
                        pGradY(i, j) = (pCenter(i + 1, j) - pCenter(i - 1, j)) / (2.0f * hy);
                    } else if (i == 0) {
                        pGradY(i, j) = (pCenter(i + 1, j) - pCenter(i, j)) / hy;  // Forward diff
                    } else {
                        pGradY(i, j) = (pCenter(i, j) - pCenter(i - 1, j)) / hy;  // Backward diff
                    }
                    
                    pGradMag(i, j) = std::sqrt(pGradX(i, j) * pGradX(i, j) + 
                                               pGradY(i, j) * pGradY(i, j));
            }
        }
    }

    // ---------------------------------------------------------
    // 4. EXPORT THERMAL DATA (CROPPED)
    // ---------------------------------------------------------
    int N_thermal = physCols - N_in_buffer - N_out_buffer;
    
    if (N_thermal <= 0) {
        std::cerr << "Error: Thermal domain size is <= 0. Check buffer sizes." << std::endl;
    } else {
        ScopedTimer t2("Output: Thermal Data Cropping & txt");
        Eigen::MatrixXf uThermal = Eigen::MatrixXf::Zero(physRows, N_thermal);
        Eigen::MatrixXf vThermal = Eigen::MatrixXf::Zero(physRows, N_thermal);
        Eigen::MatrixXf pThermal = Eigen::MatrixXf::Zero(physRows, N_thermal);
        Eigen::MatrixXf pGradThermal = Eigen::MatrixXf::Zero(physRows, N_thermal);
        
        // Slice the matrix: Skip 'N_in_buffer' columns
        for (int i = 0; i < physRows; ++i) {
            for (int j = 0; j < N_thermal; ++j) {
                int src_j = j + N_in_buffer;
                uThermal(i, j) = uCenter(i, src_j);
                vThermal(i, j) = vCenter(i, src_j);
                pThermal(i, j) = pCenter(i, src_j);
                pGradThermal(i, j) = pGradMag(i, src_j);
            }
        }
        saveMatrix(uThermal, "u_thermal");
        saveMatrix(vThermal, "v_thermal");
        saveMatrix(pThermal, "pressure_thermal");
        saveMatrix(pGradThermal, "p_gradient");
    }

    // ---------------------------------------------------------
    // 5. EXPORT FLUID VTK (FULL DOMAIN)
    // ---------------------------------------------------------
    {
        ScopedTimer t2("Output: VTK Export");
        std::string vtkFile = "ExportFiles/fluid_results.vtk";
        std::ofstream vtk(vtkFile);
        
        if (vtk.is_open()) {
            vtk << "# vtk DataFile Version 3.0\n";
            vtk << "SIMPLE CFD Results\n";
            vtk << "ASCII\n";
            vtk << "DATASET STRUCTURED_POINTS\n";
            vtk << "DIMENSIONS " << physCols << " " << physRows << " 1\n"; 
            vtk << "ORIGIN 0 0 0\n";
            vtk << "SPACING " << hx << " " << hy << " 1\n";
            vtk << "POINT_DATA " << physCols * physRows << "\n";

            // Pressure
            vtk << "SCALARS pressure double 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (int i = 0; i < physRows; ++i) {     
                for (int j = 0; j < physCols; ++j) { 
                    vtk << pCenter(i, j) << "\n";
                }
            }

            // Velocity Vectors
            vtk << "VECTORS velocity double\n";
            for (int i = 0; i < physRows; ++i) {
                for (int j = 0; j < physCols; ++j) {
                    vtk << uCenter(i, j) << " " << vCenter(i, j) << " 0.0\n";
                }
            }

            // Cell Type (Geometry) - continuous values 0=fluid, 1=solid
            vtk << "SCALARS cellType double 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (int i = 0; i < physRows; ++i) {
                for (int j = 0; j < physCols; ++j) {
                    vtk << cellType(i, j) << "\n";
                }
            }
            
            // Density field (gamma): 1=fluid, 0=solid, intermediate=buffer
            vtk << "SCALARS Density double 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (int i = 0; i < physRows; ++i) {
                for (int j = 0; j < physCols; ++j) {
                    vtk << gamma(i, j) << "\n";
                }
            }
            
            // Brinkman alpha field (penalization strength)
            vtk << "SCALARS Alpha double 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (int i = 0; i < physRows; ++i) {
                for (int j = 0; j < physCols; ++j) {
                    vtk << alpha(i, j) << "\n";
                }
            }

            // Momentum residuals (cell-centered)
            vtk << "SCALARS u_residual double 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (int i = 0; i < physRows; ++i) {
                for (int j = 0; j < physCols; ++j) {
                    vtk << uResidCenter(i, j) << "\n";
                }
            }
            vtk << "SCALARS v_residual double 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (int i = 0; i < physRows; ++i) {
                for (int j = 0; j < physCols; ++j) {
                    vtk << vResidCenter(i, j) << "\n";
                }
            }
            
            // Pressure Gradient Magnitude
            vtk << "SCALARS PressureGradient double 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (int i = 0; i < physRows; ++i) {
                for (int j = 0; j < physCols; ++j) {
                    vtk << pGradMag(i, j) << "\n";
                }
            }
            
            // Pressure Gradient Vector
            vtk << "VECTORS pressure_gradient double\n";
            for (int i = 0; i < physRows; ++i) {
                for (int j = 0; j < physCols; ++j) {
                    vtk << pGradX(i, j) << " " << pGradY(i, j) << " 0.0\n";
                }
            }
            
            std::cout << "Saved VTK to " << vtkFile << std::endl;
        }
    }

    std::cout << "Data saved to ExportFiles/" << std::endl;
}

void SIMPLE::initLogFiles(std::ofstream& residFile, std::ofstream& dpFile) {
    residFile.open("ExportFiles/residuals.txt");
    dpFile.open("ExportFiles/pressure_drop_history.txt");
    residFile << "Iter MassRMS U_RMS V_RMS MassRMSn U_RMSn V_RMSn Core_dP_AfterInletBuffer(Pa) "
              << "Full_dP_FullSystem(Pa) CFL" << std::endl;
    dpFile << "Iter Core_Total(Pa) Full_Total(Pa) Core_Static(Pa) Full_Static(Pa)" << std::endl;
    printIterationHeader();
}

void SIMPLE::printIterationHeader() const {
    std::cout << "Starting simulation..." << std::endl;
    const char* massLabel = enableNormalizedResiduals ? "MassRMSn" : "MassRMS";
    const char* uLabel = enableNormalizedResiduals ? "U-RMSn" : "U-RMS";
    const char* vLabel = enableNormalizedResiduals ? "V-RMSn" : "V-RMS";
    std::cout << std::setw(8) << "Iter"
              << std::setw(14) << massLabel
              << std::setw(14) << uLabel
              << std::setw(14) << vLabel
              << std::setw(14) << "TransRes"
              << std::setw(16) << "Core dP (Pa)"
              << std::setw(16) << "Full dP (Pa)"
              << std::setw(10) << "Time (ms)"
              << std::setw(10) << "CFL"
              << std::setw(6) << "P-It" << std::endl;
    std::cout << std::string(132, '-') << std::endl;
}

void SIMPLE::writeIterationLogs(std::ofstream& residFile,
                                std::ofstream& dpFile,
                                int iter,
                                float corePressureDrop,
                                float fullPressureDrop,
                                float coreStaticDrop,
                                float fullStaticDrop,
                                float pseudoCFL) {
    const float massRMS = residMass_RMS;
    const float uRMS = residU_RMS;
    const float vRMS = residV_RMS;
    const float massRMSn = (residMass_RMS0 > 0.0f) ? (residMass_RMS / residMass_RMS0) : 0.0f;
    const float uRMSn = (residU_RMS0 > 0.0f) ? (residU_RMS / residU_RMS0) : 0.0f;
    const float vRMSn = (residV_RMS0 > 0.0f) ? (residV_RMS / residV_RMS0) : 0.0f;

    residFile << iter << " "
              << massRMS << " "
              << uRMS << " "
              << vRMS << " "
              << massRMSn << " "
              << uRMSn << " "
              << vRMSn << " "
              << corePressureDrop << " "
              << fullPressureDrop << " "
              << pseudoCFL << std::endl;

    dpFile << iter << " "
           << corePressureDrop << " "
           << fullPressureDrop << " "
           << coreStaticDrop << " "
           << fullStaticDrop << std::endl;
}

void SIMPLE::printIterationRow(int iter,
                               float residMassVal,
                               float residUVal,
                               float residVVal,
                               float maxTransRes,
                               float corePressureDrop,
                               float fullPressureDrop,
                               float iterTimeMs,
                               int pressureIterations,
                               float pseudoCFL) const {
    std::cout << std::setw(8) << iter
              << std::setw(14) << std::scientific << std::setprecision(3) << residMassVal
              << std::setw(14) << residUVal
              << std::setw(14) << residVVal
              << std::setw(14) << maxTransRes
              << std::setw(16) << std::fixed << std::setprecision(1) << corePressureDrop
              << std::setw(16) << fullPressureDrop
              << std::setw(10) << std::fixed << std::setprecision(1) << iterTimeMs
              << std::setw(10) << std::fixed << std::setprecision(2) << pseudoCFL
              << std::setw(6) << pressureIterations
              << std::endl;
}

void SIMPLE::printStaticDp(int iter,
                           float coreStaticDrop,
                           float fullStaticDrop) const {
    std::cout << "         Static dP (Core/Full): " 
              << std::setw(12) << std::fixed << std::setprecision(1) << coreStaticDrop << " / "
              << std::setw(12) << fullStaticDrop << " Pa"
              << std::endl;
}

void SIMPLE::paintBoundaries() {
    const int physRows = M - 1;
    const int physCols = N - 1;
    Eigen::MatrixXf BCs = Eigen::MatrixXf::Zero(physRows, physCols);
    for (int i = 0; i < physRows; ++i) {
        for (int j = 0; j < physCols; ++j) {
            BCs(i, j) = checkBoundaries(i + 1, j + 1);
        }
    }
    saveMatrix(BCs, "BC");
}

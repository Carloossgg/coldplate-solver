@echo off
REM ============================================================================
REM Build script for SIMPLE CFD Solver (Windows)
REM ============================================================================

echo.
echo ========================================================================
echo              Building SIMPLE CFD Solver with AMGCL
echo ========================================================================
echo.

REM Check if g++ is available
where g++ >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: g++ not found in PATH!
    echo Please install MinGW-w64 or add g++ to your PATH.
    pause
    exit /b 1
)

REM Display g++ version
echo Compiler version:
g++ --version | findstr "g++"
echo.

REM Clean old executable
if exist simple.exe (
    echo Removing old executable...
    del /f simple.exe
)

echo Compiling with full optimization and AMGCL support...
echo.

REM Compile with all features (AMGCL_NO_BOOST removes Boost dependency)
g++ -std=c++17 -O3 -mavx2 -mfma -ftree-vectorize -fopenmp -DUSE_AMGCL -DAMGCL_NO_BOOST -I. -I"ThermalSolver/amgcl" -I"Utilities/eigen-3.4.0" SIMPLE.cpp Utilities/*.cpp Utilities\solvers\cpu\*.cpp -o simple.exe

if %errorlevel% neq 0 (
    echo.
    echo ========================================================================
    echo                          BUILD FAILED
    echo ========================================================================
    echo.
    echo Check the error messages above.
    echo See BUILD_INSTRUCTIONS.md for help.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo                        BUILD SUCCESSFUL!
echo ========================================================================
echo   Executable: simple.exe
echo   AMGCL: ENABLED (pressureSolverType=2 available)
echo   OpenMP: ENABLED (multi-threading active)
echo ========================================================================
echo.

echo To run the solver, type: simple.exe
echo.
pause


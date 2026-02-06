@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo       Building SIMPLE CFD Solver with AMGCL-CUDA GPU Support
echo ========================================================================
echo.

REM Check for CUDA
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found! Please install CUDA Toolkit and add to PATH.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    goto :error
)

REM Set paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set AMGCL_PATH=ThermalSolver\amgcl
set EIGEN_PATH=Utilities\eigen-3.4.0
set BOOST_PATH=C:\msys64\mingw64\include

REM Initialize Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

echo Removing old files...
if exist simple_gpu.exe del simple_gpu.exe
if exist *.obj del *.obj

echo.
echo Step 1: Compiling CUDA solvers with nvcc...
echo.

REM Compile pressure solver
nvcc -c Utilities\solvers\gpu\pressure_solver.cu ^
    -o pressure_solver_gpu.obj ^
    -O3 ^
    -std=c++17 ^
    -I"%AMGCL_PATH%" ^
    -I"%EIGEN_PATH%" ^
    -I"%BOOST_PATH%" ^
    -I"%CUDA_PATH%\include" ^
    -IUtilities\solvers\gpu ^
    -Xcompiler "/EHsc /O2 /std:c++17" ^
    --expt-relaxed-constexpr ^
    -use_fast_math ^
    -arch=sm_89 ^
    -DUSE_AMGCL_CUDA

if errorlevel 1 (
    echo.
    echo ERROR: CUDA pressure_solver compilation failed!
    goto :error
)

REM Compile momentum solver
nvcc -c Utilities\solvers\gpu\momentum_solver.cu ^
    -o momentum_solver_gpu.obj ^
    -O3 ^
    -std=c++17 ^
    -I"%AMGCL_PATH%" ^
    -I"%EIGEN_PATH%" ^
    -I"%BOOST_PATH%" ^
    -I"%CUDA_PATH%\include" ^
    -IUtilities\solvers\gpu ^
    -Xcompiler "/EHsc /O2 /std:c++17" ^
    --expt-relaxed-constexpr ^
    -use_fast_math ^
    -arch=sm_89 ^
    -DUSE_AMGCL_CUDA

if errorlevel 1 (
    echo.
    echo ERROR: CUDA momentum_solver compilation failed!
    goto :error
)

echo CUDA solvers compiled successfully.
echo.
echo Step 2: Compiling C++ sources with MSVC...
echo.

cl /c /EHsc /O2 /arch:AVX2 /fp:fast /openmp:llvm /std:c++17 ^
    /I. ^
    /I"%EIGEN_PATH%" ^
    /I"%AMGCL_PATH%" ^
    /I"%BOOST_PATH%" ^
    /I"%CUDA_PATH%\include" ^
    /DUSE_AMGCL /DUSE_AMGCL_CUDA ^
    SIMPLE.cpp ^
    Utilities\boundaries.cpp ^
    Utilities\convection.cpp ^
    Utilities\iterations.cpp ^
    Utilities\momentum_solver.cpp ^
    Utilities\output.cpp ^
    Utilities\postprocessing.cpp ^
    Utilities\pressure_solver.cpp ^
    Utilities\stabilization.cpp ^
    Utilities\time_control.cpp ^
    Utilities\solvers\cpu\pcg_solver.cpp ^
    Utilities\solvers\cpu\jacobi_solver.cpp

if errorlevel 1 (
    echo.
    echo ERROR: C++ compilation failed!
    goto :error
)

echo C++ sources compiled successfully.
echo.
echo Step 3: Linking...
echo.

link /OUT:simple_gpu.exe ^
    /LIBPATH:"%CUDA_PATH%\lib\x64" ^
    cudart.lib cusparse.lib cublas.lib ^
    SIMPLE.obj ^
    boundaries.obj ^
    convection.obj ^
    iterations.obj ^
    momentum_solver.obj ^
    output.obj ^
    postprocessing.obj ^
    pressure_solver.obj ^
    stabilization.obj ^
    time_control.obj ^
    pcg_solver.obj ^
    pressure_solver_gpu.obj ^
    momentum_solver_gpu.obj ^
    jacobi_solver.obj

if errorlevel 1 (
    echo.
    echo ERROR: Linking failed!
    goto :error
)

echo.
echo ========================================================================
echo                   BUILD SUCCESSFUL (AMGCL-CUDA GPU)
echo ========================================================================
echo.
echo   Executable: simple_gpu.exe
echo   GPU Solver: AMGCL-CUDA (pressureSolverType = 4)
echo.
echo   Single precision (float) only - 2x faster!
echo   Expected: Same convergence (10-15 iterations), faster speed.
echo.
echo ========================================================================
goto :end

:error
echo.
echo ========================================================================
echo                        BUILD FAILED
echo ========================================================================
echo Check the error messages above.

:end
REM Cleanup object files
del *.obj 2>nul

pause

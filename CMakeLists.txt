# =============================================================================
# Matrix Lie Groups — CMake Build Configuration
# =============================================================================
cmake_minimum_required(VERSION 3.20)

project(matrix_groups
    VERSION 2.0.0
    DESCRIPTION "Computational kernel for matrix Lie groups: SO(n), SE(3), SymPD"
    LANGUAGES CXX CUDA
)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Eigen 3.4+ (header-only linear algebra — REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")

# Google Test (optional — enables test suite)
find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "GTest found — tests will be built")
else()
    message(STATUS "GTest NOT found — skipping test target")
endif()

# PyTorch C++ API (optional — enables torch bindings)
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "PyTorch found: ${Torch_DIR}")
else()
    message(STATUS "PyTorch NOT found — skipping CUDA extension target")
endif()

# ---------------------------------------------------------------------------
# Core library (header-only interface)
# ---------------------------------------------------------------------------
# All implementations live in include/mg/*.hpp as inline / template code.
# No compiled sources are needed for the core library itself.
add_library(mg_core INTERFACE)
target_include_directories(mg_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(mg_core INTERFACE Eigen3::Eigen)

# ---------------------------------------------------------------------------
# Compile options shared across targets
# ---------------------------------------------------------------------------
# Expose as a helper so test / extension targets can reuse it.
add_library(mg_compile_options INTERFACE)
target_compile_options(mg_compile_options INTERFACE
    -O3 -Wall -Wextra -Wpedantic
    $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>
    $<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>
)
target_compile_features(mg_compile_options INTERFACE cxx_std_20)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(GTest_FOUND)
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# PyTorch CUDA extension
# ---------------------------------------------------------------------------
if(Torch_FOUND AND CMAKE_CUDA_COMPILER)
    add_library(matrix_groups_cuda SHARED src/torch_bindings.cpp)
    target_link_libraries(matrix_groups_cuda
        PRIVATE
            mg_core
            ${TORCH_LIBRARIES}
    )
    target_include_directories(matrix_groups_cuda PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(matrix_groups_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    # Suppress PyTorch's noisy deprecation warnings during our build
    target_compile_options(matrix_groups_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
    )
    # Torch extensions convention: no "lib" prefix on shared libraries
    set_target_properties(matrix_groups_cuda PROPERTIES
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS mg_core
    EXPORT matrix_groups_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/mg/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mg
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT matrix_groups_targets
    FILE matrix_groupsTargets.cmake
    NAMESPACE matrix_groups::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrix_groups
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== Matrix Lie Groups Build Summary ===")
message(STATUS "  Version            : ${PROJECT_VERSION}")
message(STATUS "  C++ standard       : ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard      : ${CMAKE_CUDA_STANDARD}")
message(STATUS "  Eigen3             : ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  GTest              : ${GTest_FOUND}")
message(STATUS "  PyTorch            : ${Torch_FOUND}")
message(STATUS "=====================================")
message(STATUS "")

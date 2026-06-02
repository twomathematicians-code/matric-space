# Tests CMakeLists.txt
if(NOT GTest_FOUND)
    return()
endif()

enable_testing()

add_executable(test_matrix_groups
    test_so3.cpp
    test_se3.cpp
    test_sympd.cpp
    test_so_n.cpp
)
target_link_libraries(test_matrix_groups
    PRIVATE mg_core GTest::gtest GTest::gtest_main
)
target_include_directories(test_matrix_groups PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(test_matrix_groups)

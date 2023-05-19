include(CTest)
enable_testing()

add_executable(litchi_test test.cc)

set_target_properties(litchi_test PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(litchi_test
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

gtest_discover_tests(litchi_test)
add_custom_target(run_tests COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS litchi_test)
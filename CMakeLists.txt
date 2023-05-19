#####
# Set minimum version of CMake.
CMAKE_MINIMUM_REQUIRED(VERSION 3.2 FATAL_ERROR)

#####
# Set variables:
project(litchi LANGUAGES C CXX)

#####
# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("==========================================================================================")
        message(STATUS "Build type: Debug. Performance will be terrible!")
        message(STATUS "Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
        message("==========================================================================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

#####
# Define user options

option(BUILD_TESTS "Set to On to build tests" ON)

#####
# Create the library target
set(project_library_target_name ${PROJECT_NAME})
set(PACKAGE_NAME Litchi)

add_library(${project_library_target_name} INTERFACE)

#####
# Setup the optional dependencies

# Find GTest
find_package(GTest MODULE REQUIRED)
message(STATUS "Found GTest: ${GTEST_INCLUDE_DIR}")
list(APPEND REQUIRED_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

#####
# Setup the compiler options

# set C++ standard to C++14.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++14 support has been enabled by default.")

# include extra flags to the compiler
set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing -Wno-deprecated")
set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pthread")

#####
# Set compiler options
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")

#####
# Setup the cmake config files
string(REGEX REPLACE "_" "" PROJECT_NAME_JOINED ${PROJECT_NAME})

# Set up install directories. INCLUDE_INSTALL_DIR must
# not be absolute paths.
set(include_install_dir include)
set(include_install_dir_postfix "${project_library_target_name}")
set(include_install_dir_full    "${include_install_dir}/${include_install_dir_postfix}")

# configure the library target
target_include_directories(
    ${project_library_target_name} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${include_install_dir_full}>
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")

# Install headers
install(DIRECTORY   ${PROJECT_SOURCE_DIR}/${project_library_target_name}
        DESTINATION ${include_install_dir})

if(BUILD_TESTS)
    add_subdirectory(test)
endif(BUILD_TESTS)
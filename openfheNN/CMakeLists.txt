cmake_minimum_required(VERSION 3.16.3)
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

### To use gcc/g++ on a Macintosh, you must set the Compilers
### here, not inside the project
##if(APPLE)
##       set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
##       set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")
##endif()
### TODO: for now, we use CLang for Mac
###
### In order to create OpenFHE's static libraries you should enable
### the BUILD_STATIC option. For that, you run "cmake .. -DBUILD_STATIC=ON".
### After having your link completed you will find static libs
### with the suffix "_static" in ./build/libs/.
### Examples: OPENFHEpke_static.a, OPENFHEcore_static.a, etc.
### After you run "make install" in your build directory, you can build your custom application.
### If you need your application to be linked statically, then run "cmake .. -DBUILD_STATIC=ON"

project(demo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)

find_package(OpenFHE CONFIG REQUIRED)
if(OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

if(UNIX AND NOT APPLE)
    add_link_options(-Wl,--no-as-needed)
endif()

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})
if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_library(mainlib_common
    ${PROJECT_ROOT}/src/common/utils_ckks.cpp
    ${PROJECT_ROOT}/src/common/campaign_helper.cpp
    ${PROJECT_ROOT}/src/common/campaign_logger.cpp
    ${PROJECT_ROOT}/src/common/campaign_registry.cpp
)


add_library(backend_openfhe
    ${PROJECT_ROOT}/backends/openfhe/src/backend.cpp
    ${PROJECT_ROOT}/backends/openfheNN/src/utils_nn.cpp
)

target_include_directories(backend_openfhe PUBLIC
    ${PROJECT_ROOT}/src/common
    ${PROJECT_ROOT}/backends/openfhe/src
    ${PROJECT_ROOT}/backends/openfhe/src/nn_mnist/src
    ${OPENFHE_INCLUDE_DIR}  # donde tengas OpenFHE
)
target_link_libraries(backend_openfhe
    PUBLIC
        mainlib_common
        ${OpenFHE_SHARED_LIBRARIES}
)
set(EXECUTABLES
    nn_mnist
    randomSingleBitFlip
)


foreach(exec_name ${EXECUTABLES})
    add_executable(${exec_name} src/${exec_name}.cpp)
    target_link_libraries(${exec_name} PRIVATE mainlib_common backend_openfhe ${OpenFHE_SHARED_LIBRARIES})
endforeach()



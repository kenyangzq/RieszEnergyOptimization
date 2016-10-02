cmake_minimum_required(VERSION 2.8)
project(ExamplesNumericalSolvers)
#
# required libraries: libeigen3, libgtest
#
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
#set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

# check existence of c++11 compiler
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#macro(remove_cxx_flag flag)
    #string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
#endmacro()

#message(${CMAKE_CXX_FLAGS_RELEASE}) # print "-O3 -DNDEBUG"
#remove_cxx_flag("-O3")
#message(${CMAKE_CXX_FLAGS_RELEASE}) # print "-DNDEBUG"
#set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
#message(${CMAKE_CXX_FLAGS_RELEASE}) 


#find_package(OpenMP REQUIRED)
#if(OPENMP_FOUND)
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

# Eigen3 Headerfiles
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
find_package(Eigen3 REQUIRED)
include_directories(EIGEN3_INCLUDE_DIR)


# #GTest headerfiles
#find_package(GTest REQUIRED)
#include_directories(GTest_INCLUDE_DIR)
#include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
#include_directories(/home/alex/Recherche/Code/CppNumericalSolvers/include/cppoptlib)

include_directories(./include/cppoptlib)
include_directories(./include/eigen)


#set(Boost_USE_STATIC_LIBS ON) 
##set(Boost_USE_MULTITHREADED ON)  
###set(Boost_USE_STATIC_RUNTIME OFF) 
#find_package(Boost REQUIRED COMPONENTS system filesystem regex ) 
#include_directories(${Boost_INCLUDE_DIRS}) 

#if(Boost_FOUND)
    #add_executable(progname file1.cxx file2.cxx) 
    #target_link_libraries(progname ${Boost_LIBRARIES})
#endif()



SET( RIESZ_FILES main ) 
#set( SOURCES min_energy.h)



#set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} ) #/bin/examples
#foreach( currentfile ${RIESZ_FILES} )
    #add_executable( ${currentfile} "${currentfile}.cpp" ${SOURCES} )
    #target_link_libraries(${currentfile} ${GTEST_LIBRARIES}
        #${GTEST_MAIN_LIBRARIES}  ${Boost_LIBRARIES} pthread )
#endforeach( currentfile ${RIESZ_FILES} )


file(GLOB SOURCES
    *.h
    main.cpp
    L-BFGS_truncated.cpp
)
add_executable( main ${SOURCES})
target_link_libraries(${currentfile} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES}  ${Boost_LIBRARIES} pthread )


cmake_minimum_required (VERSION 3.0)
project (ET_Framework)

find_package (OpenCV REQUIRED)

add_executable (framework src/main.cxx)
target_link_libraries (framework ${OpenCV_LIBS})

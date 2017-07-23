cmake_minimum_required (VERSION 3.5)
project (Raspivid_Client_Framework)

find_package (OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})

add_executable (raspivid_client_framework src/main.cpp)
target_link_libraries (raspivid_client_framework ${OpenCV_LIBS})

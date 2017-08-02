# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/build"

# Include any dependencies generated for this target.
include CMakeFiles/raspivid_client_framework.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/raspivid_client_framework.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/raspivid_client_framework.dir/flags.make

CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o: CMakeFiles/raspivid_client_framework.dir/flags.make
CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o -c "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/src/main.cpp"

CMakeFiles/raspivid_client_framework.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/raspivid_client_framework.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/src/main.cpp" > CMakeFiles/raspivid_client_framework.dir/src/main.cpp.i

CMakeFiles/raspivid_client_framework.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/raspivid_client_framework.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/src/main.cpp" -o CMakeFiles/raspivid_client_framework.dir/src/main.cpp.s

CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o.requires

CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o.provides: CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/raspivid_client_framework.dir/build.make CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o.provides

CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o.provides.build: CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o


# Object files for target raspivid_client_framework
raspivid_client_framework_OBJECTS = \
"CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o"

# External object files for target raspivid_client_framework
raspivid_client_framework_EXTERNAL_OBJECTS =

raspivid_client_framework: CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o
raspivid_client_framework: CMakeFiles/raspivid_client_framework.dir/build.make
raspivid_client_framework: /usr/local/lib/libopencv_dnn.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_ml.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_objdetect.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_shape.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_stitching.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_superres.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_videostab.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_calib3d.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_features2d.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_flann.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_highgui.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_photo.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_video.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_videoio.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_imgproc.so.3.3.0
raspivid_client_framework: /usr/local/lib/libopencv_core.so.3.3.0
raspivid_client_framework: CMakeFiles/raspivid_client_framework.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable raspivid_client_framework"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/raspivid_client_framework.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/raspivid_client_framework.dir/build: raspivid_client_framework

.PHONY : CMakeFiles/raspivid_client_framework.dir/build

CMakeFiles/raspivid_client_framework.dir/requires: CMakeFiles/raspivid_client_framework.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/raspivid_client_framework.dir/requires

CMakeFiles/raspivid_client_framework.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/raspivid_client_framework.dir/cmake_clean.cmake
.PHONY : CMakeFiles/raspivid_client_framework.dir/clean

CMakeFiles/raspivid_client_framework.dir/depend:
	cd "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework" "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework" "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/build" "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/build" "/home/claudio/Projects/Eye Tracker/Repo/Eye-Tracker/Raspi to OpenCV/Framework/build/CMakeFiles/raspivid_client_framework.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/raspivid_client_framework.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alan/dev/optix/SDK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alan/dev/optix/SDK

# Include any dependencies generated for this target.
include support/imgui/CMakeFiles/imgui.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include support/imgui/CMakeFiles/imgui.dir/compiler_depend.make

# Include the progress variables for this target.
include support/imgui/CMakeFiles/imgui.dir/progress.make

# Include the compile flags for this target's objects.
include support/imgui/CMakeFiles/imgui.dir/flags.make

support/imgui/CMakeFiles/imgui.dir/imgui.cpp.o: support/imgui/CMakeFiles/imgui.dir/flags.make
support/imgui/CMakeFiles/imgui.dir/imgui.cpp.o: support/imgui/imgui.cpp
support/imgui/CMakeFiles/imgui.dir/imgui.cpp.o: support/imgui/CMakeFiles/imgui.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/alan/dev/optix/SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui.cpp.o"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT support/imgui/CMakeFiles/imgui.dir/imgui.cpp.o -MF CMakeFiles/imgui.dir/imgui.cpp.o.d -o CMakeFiles/imgui.dir/imgui.cpp.o -c /home/alan/dev/optix/SDK/support/imgui/imgui.cpp

support/imgui/CMakeFiles/imgui.dir/imgui.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgui.dir/imgui.cpp.i"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alan/dev/optix/SDK/support/imgui/imgui.cpp > CMakeFiles/imgui.dir/imgui.cpp.i

support/imgui/CMakeFiles/imgui.dir/imgui.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgui.dir/imgui.cpp.s"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alan/dev/optix/SDK/support/imgui/imgui.cpp -o CMakeFiles/imgui.dir/imgui.cpp.s

support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.o: support/imgui/CMakeFiles/imgui.dir/flags.make
support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.o: support/imgui/imgui_demo.cpp
support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.o: support/imgui/CMakeFiles/imgui.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/alan/dev/optix/SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.o"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.o -MF CMakeFiles/imgui.dir/imgui_demo.cpp.o.d -o CMakeFiles/imgui.dir/imgui_demo.cpp.o -c /home/alan/dev/optix/SDK/support/imgui/imgui_demo.cpp

support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgui.dir/imgui_demo.cpp.i"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alan/dev/optix/SDK/support/imgui/imgui_demo.cpp > CMakeFiles/imgui.dir/imgui_demo.cpp.i

support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgui.dir/imgui_demo.cpp.s"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alan/dev/optix/SDK/support/imgui/imgui_demo.cpp -o CMakeFiles/imgui.dir/imgui_demo.cpp.s

support/imgui/CMakeFiles/imgui.dir/imgui_draw.cpp.o: support/imgui/CMakeFiles/imgui.dir/flags.make
support/imgui/CMakeFiles/imgui.dir/imgui_draw.cpp.o: support/imgui/imgui_draw.cpp
support/imgui/CMakeFiles/imgui.dir/imgui_draw.cpp.o: support/imgui/CMakeFiles/imgui.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/alan/dev/optix/SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui_draw.cpp.o"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT support/imgui/CMakeFiles/imgui.dir/imgui_draw.cpp.o -MF CMakeFiles/imgui.dir/imgui_draw.cpp.o.d -o CMakeFiles/imgui.dir/imgui_draw.cpp.o -c /home/alan/dev/optix/SDK/support/imgui/imgui_draw.cpp

support/imgui/CMakeFiles/imgui.dir/imgui_draw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgui.dir/imgui_draw.cpp.i"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alan/dev/optix/SDK/support/imgui/imgui_draw.cpp > CMakeFiles/imgui.dir/imgui_draw.cpp.i

support/imgui/CMakeFiles/imgui.dir/imgui_draw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgui.dir/imgui_draw.cpp.s"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alan/dev/optix/SDK/support/imgui/imgui_draw.cpp -o CMakeFiles/imgui.dir/imgui_draw.cpp.s

support/imgui/CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.o: support/imgui/CMakeFiles/imgui.dir/flags.make
support/imgui/CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.o: support/imgui/imgui_impl_glfw.cpp
support/imgui/CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.o: support/imgui/CMakeFiles/imgui.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/alan/dev/optix/SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.o"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT support/imgui/CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.o -MF CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.o.d -o CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.o -c /home/alan/dev/optix/SDK/support/imgui/imgui_impl_glfw.cpp

support/imgui/CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.i"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alan/dev/optix/SDK/support/imgui/imgui_impl_glfw.cpp > CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.i

support/imgui/CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.s"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alan/dev/optix/SDK/support/imgui/imgui_impl_glfw.cpp -o CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.s

support/imgui/CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.o: support/imgui/CMakeFiles/imgui.dir/flags.make
support/imgui/CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.o: support/imgui/imgui_impl_opengl3.cpp
support/imgui/CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.o: support/imgui/CMakeFiles/imgui.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/alan/dev/optix/SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.o"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT support/imgui/CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.o -MF CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.o.d -o CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.o -c /home/alan/dev/optix/SDK/support/imgui/imgui_impl_opengl3.cpp

support/imgui/CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.i"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alan/dev/optix/SDK/support/imgui/imgui_impl_opengl3.cpp > CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.i

support/imgui/CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.s"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alan/dev/optix/SDK/support/imgui/imgui_impl_opengl3.cpp -o CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.s

support/imgui/CMakeFiles/imgui.dir/imgui_widgets.cpp.o: support/imgui/CMakeFiles/imgui.dir/flags.make
support/imgui/CMakeFiles/imgui.dir/imgui_widgets.cpp.o: support/imgui/imgui_widgets.cpp
support/imgui/CMakeFiles/imgui.dir/imgui_widgets.cpp.o: support/imgui/CMakeFiles/imgui.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/alan/dev/optix/SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui_widgets.cpp.o"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT support/imgui/CMakeFiles/imgui.dir/imgui_widgets.cpp.o -MF CMakeFiles/imgui.dir/imgui_widgets.cpp.o.d -o CMakeFiles/imgui.dir/imgui_widgets.cpp.o -c /home/alan/dev/optix/SDK/support/imgui/imgui_widgets.cpp

support/imgui/CMakeFiles/imgui.dir/imgui_widgets.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgui.dir/imgui_widgets.cpp.i"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alan/dev/optix/SDK/support/imgui/imgui_widgets.cpp > CMakeFiles/imgui.dir/imgui_widgets.cpp.i

support/imgui/CMakeFiles/imgui.dir/imgui_widgets.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgui.dir/imgui_widgets.cpp.s"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alan/dev/optix/SDK/support/imgui/imgui_widgets.cpp -o CMakeFiles/imgui.dir/imgui_widgets.cpp.s

support/imgui/CMakeFiles/imgui.dir/imgui_stdlib.cpp.o: support/imgui/CMakeFiles/imgui.dir/flags.make
support/imgui/CMakeFiles/imgui.dir/imgui_stdlib.cpp.o: support/imgui/imgui_stdlib.cpp
support/imgui/CMakeFiles/imgui.dir/imgui_stdlib.cpp.o: support/imgui/CMakeFiles/imgui.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/alan/dev/optix/SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui_stdlib.cpp.o"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT support/imgui/CMakeFiles/imgui.dir/imgui_stdlib.cpp.o -MF CMakeFiles/imgui.dir/imgui_stdlib.cpp.o.d -o CMakeFiles/imgui.dir/imgui_stdlib.cpp.o -c /home/alan/dev/optix/SDK/support/imgui/imgui_stdlib.cpp

support/imgui/CMakeFiles/imgui.dir/imgui_stdlib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgui.dir/imgui_stdlib.cpp.i"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alan/dev/optix/SDK/support/imgui/imgui_stdlib.cpp > CMakeFiles/imgui.dir/imgui_stdlib.cpp.i

support/imgui/CMakeFiles/imgui.dir/imgui_stdlib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgui.dir/imgui_stdlib.cpp.s"
	cd /home/alan/dev/optix/SDK/support/imgui && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alan/dev/optix/SDK/support/imgui/imgui_stdlib.cpp -o CMakeFiles/imgui.dir/imgui_stdlib.cpp.s

# Object files for target imgui
imgui_OBJECTS = \
"CMakeFiles/imgui.dir/imgui.cpp.o" \
"CMakeFiles/imgui.dir/imgui_demo.cpp.o" \
"CMakeFiles/imgui.dir/imgui_draw.cpp.o" \
"CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.o" \
"CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.o" \
"CMakeFiles/imgui.dir/imgui_widgets.cpp.o" \
"CMakeFiles/imgui.dir/imgui_stdlib.cpp.o"

# External object files for target imgui
imgui_EXTERNAL_OBJECTS =

lib/libimgui.a: support/imgui/CMakeFiles/imgui.dir/imgui.cpp.o
lib/libimgui.a: support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.o
lib/libimgui.a: support/imgui/CMakeFiles/imgui.dir/imgui_draw.cpp.o
lib/libimgui.a: support/imgui/CMakeFiles/imgui.dir/imgui_impl_glfw.cpp.o
lib/libimgui.a: support/imgui/CMakeFiles/imgui.dir/imgui_impl_opengl3.cpp.o
lib/libimgui.a: support/imgui/CMakeFiles/imgui.dir/imgui_widgets.cpp.o
lib/libimgui.a: support/imgui/CMakeFiles/imgui.dir/imgui_stdlib.cpp.o
lib/libimgui.a: support/imgui/CMakeFiles/imgui.dir/build.make
lib/libimgui.a: support/imgui/CMakeFiles/imgui.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/alan/dev/optix/SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX static library ../../lib/libimgui.a"
	cd /home/alan/dev/optix/SDK/support/imgui && $(CMAKE_COMMAND) -P CMakeFiles/imgui.dir/cmake_clean_target.cmake
	cd /home/alan/dev/optix/SDK/support/imgui && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imgui.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
support/imgui/CMakeFiles/imgui.dir/build: lib/libimgui.a
.PHONY : support/imgui/CMakeFiles/imgui.dir/build

support/imgui/CMakeFiles/imgui.dir/clean:
	cd /home/alan/dev/optix/SDK/support/imgui && $(CMAKE_COMMAND) -P CMakeFiles/imgui.dir/cmake_clean.cmake
.PHONY : support/imgui/CMakeFiles/imgui.dir/clean

support/imgui/CMakeFiles/imgui.dir/depend:
	cd /home/alan/dev/optix/SDK && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alan/dev/optix/SDK /home/alan/dev/optix/SDK/support/imgui /home/alan/dev/optix/SDK /home/alan/dev/optix/SDK/support/imgui /home/alan/dev/optix/SDK/support/imgui/CMakeFiles/imgui.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : support/imgui/CMakeFiles/imgui.dir/depend


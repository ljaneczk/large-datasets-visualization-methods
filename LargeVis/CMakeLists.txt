cmake_minimum_required(VERSION 3.9)
project(LargeVis)

set(CMAKE_CXX_STANDARD 11)

add_executable(LargeVis
        cmake-build-debug/cmake_install.cmake
        cmake-build-debug/CMakeCache.txt
        cmake-build-debug/LargeVis.cbp
        cmake-build-debug/Makefile
        Examples/CondMat/CondMat_network.txt
        Examples/CondMat/CondMat_plot.png
        Examples/CondMat/CondMat_vec2D.txt
        Examples/CondMat/run_CondMat.sh
        Examples/MNIST/mnist_label.txt
        Examples/MNIST/mnist_plot.png
        Examples/MNIST/mnist_vec2D.txt
        Examples/MNIST/mnist_vec784D.txt.zip
        Examples/MNIST/run_mnist.sh
        Linux/ANNOY/annoylib.h
        Linux/ANNOY/kissrandom.h
        Linux/build/lib.macosx-10.11-intel-2.7/LargeVis.so
        Linux/build/temp.linux-x86_64-3.6/LargeVis.o
        Linux/build/temp.macosx-10.11-intel-2.7/LargeVis.o
        Linux/build/temp.macosx-10.11-intel-2.7/LargeVismodule.o
        Linux/irun.sh
        Linux/LargeVis
        Linux/LargeVis.cpp
        Linux/LargeVis.h
        Linux/LargeVismodule.cpp
        Linux/main.cpp
        Linux/setup.py
        Windows/ANNOY/annoylib.h
        Windows/ANNOY/kissrandom.h
        Windows/ANNOY/mman.cpp
        Windows/ANNOY/mman.h
        Windows/LargeVis.cpp
        Windows/LargeVis.h
        Windows/LargeVismodule.cpp
        Windows/main.cpp
        Windows/setup.py
        CMakeLists.txt
        LargeVis_run.py
        LICENSE
        plot.py
        README.md)

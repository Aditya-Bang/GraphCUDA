cmake_minimum_required(VERSION 3.14)
project(example LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(example example.cpp)

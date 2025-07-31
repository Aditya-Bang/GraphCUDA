# GraphCUDA

Todos:

1. check if dlls work on windows for cuda (dependancy inspector + check torch/lib) (setuptools imports need torch+cu12.4?)
2. add support for adding dll directory in init.py based on os
3. figure out how to expose classes in pybind11 + testing, and import all classes/funcs in one command with ns
4. cudaextension setup.py
5. Test running a program with cublas, and checking if it works with cuda extension.
6. actually make optimized gcn + relu (figure out architecture, maybe only gcn layer classes and training loop in python/how to merge with pytorch)
7. proper documentation check all readmes


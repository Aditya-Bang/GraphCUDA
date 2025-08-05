# GraphCUDA

Todos:

1. get initial cuda kernel up and running
2. move cpp code to /, and move all other code to old/ folder, or better name
3. look at maxk gnn fast cuda implementation
4. look at gnn advisor
5. create wavelet gnn
6. run tests against pytorch-native implementation (not pygeometric), both sparse and non sparse mm


1. check if dlls work on windows for cuda (dependancy inspector + check torch/lib) (setuptools imports need torch+cu12.4?)
2. add support for adding dll directory in init.py based on os
3. figure out how to expose classes in pybind11 + testing, and import all classes/funcs in one command with ns
4. cudaextension setup.py
5. Test running a program with cublas, and checking if it works with cuda extension.
6. actually make optimized gcn + relu (figure out architecture, maybe only gcn layer classes and training loop in python/how to merge with pytorch)
7. proper documentation check all readmes


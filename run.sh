# only build CUDA extension and 
rm -rf build
rm *.so
python setup.py build_ext
python setup.py install
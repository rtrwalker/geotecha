cls
pip uninstall geotecha
del build
python setup.py build --compiler=mingw32
python setup.py install
nosetests geotecha -v -w C:\Anaconda3\envs\py34\Lib\site-packages\ --with-doctest --doctest-options=+ELLIPSIS --verbosity=3
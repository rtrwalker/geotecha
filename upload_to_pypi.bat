rem Upload python package to pypi
rem Rohan Walker
rem Feb 2015
rem
@rem
rem py27
rem ************************************************
C:\Anaconda3\envs\py27\python.exe setup.py register -r https://testpypi.python.org/pypi
rem ************************************************
C:\Anaconda3\envs\py27\python.exe setup.py sdist upload -r https://testpypi.python.org/pypi
rem ************************************************
C:\Anaconda3\envs\py27\python.exe setup.py bdist_egg upload -r https://testpypi.python.org/pypi
rem ************************************************
C:\Anaconda3\envs\py27\python.exe setup.py bdist_wininst upload -r https://testpypi.python.org/pypi
rem ************************************************
C:\Anaconda3\envs\py27\python.exe setup.py upload_docs -r https://testpypi.python.org/pypi --upload-dir=docs/_build/html
@rem Note that upload_sphinx does not work yet on Python34 (12-Feb-2015)
@rem may need --target-version=2.7 --quiet
rem py34
rem ************************************************
C:\Anaconda3\envs\py34\python.exe setup.py bdist_egg upload -r https://testpypi.python.org/pypi
rem ************************************************
C:\Anaconda3\envs\py34\python.exe setup.py bdist_wininst upload -r https://testpypi.python.org/pypi
rem ************************************************
pause




rem Upload python package to pypi
rem Rohan Walker
rem June 3 2016
::
:: ************************************************
:: You will need have a ".pypirc" file located in your home directory (e.g.
:: "C:\Users\rohanw\.pypirc").
:: -Create a new text file save as "pypirc.txt" or something then in windows
::  explorer rename it ".pypirc." (note if you don't have the trailing "."
::  you will get an error).  Make sure you can see the file extensions (Start
::  menu; type "folder options"; on the view tab unckeck "Hide extensions for
::  known file types).  Add the following to the pypirc file (without leading "::"):
::[distutils]
::index-servers=
::    pypi
::    test
::
::[test]
::repository = https://testpypi.python.org/pypi
::username = rtrwalker
::password = <password>
::[pypi]
::repository = https://pypi.python.org/pypi
::username = rtrwalker
::password = <password>
:: ************************************************
::
:: You might want to make the temporary changes to the bat file code below
:: depending on where you run it:
:: - Once you are ready to upload to pypi rather that the test site perform
::   a find and replace, replacing "testpypi" with "pypi".
:: - Your python envs may have different names than py27 and py34. Change
::   as required.
:: - Your anaconda envs may be in different location.  change as required.
:: - You only need to register, upload source, and upload docs once.  If
:: - Uploading different distributions just comment out i.e. "::" the
::   relevant lines
rem
::
rem py27
rem ************************************************
::
C:\Anaconda3\envs\py27\python.exe setup.py register -r https://testpypi.python.org/pypi
::
rem ************************************************
::
C:\Anaconda3\envs\py27\python.exe setup.py sdist upload -r https://testpypi.python.org/pypi
::
rem ************************************************
::
C:\Anaconda3\envs\py27\python.exe setup.py bdist_egg upload -r https://testpypi.python.org/pypi
::
rem ************************************************
::
C:\Anaconda3\envs\py27\python.exe setup.py bdist_wininst upload -r https://testpypi.python.org/pypi
::
rem ************************************************
::
C:\Anaconda3\envs\py27\python.exe setup.py upload_docs -r https://testpypi.python.org/pypi --upload-dir=docs/_build/html
:: Note that upload_sphinx does not work yet on Python34 (12-Feb-2015)
:: may need --target-version=2.7 --quiet
::
rem py34
rem ************************************************
::
C:\Anaconda3\envs\py34\python.exe setup.py bdist_egg upload -r https://testpypi.python.org/pypi
::
rem ************************************************
::
C:\Anaconda3\envs\py34\python.exe setup.py bdist_wininst upload -r https://testpypi.python.org/pypi
::
rem ************************************************
pause




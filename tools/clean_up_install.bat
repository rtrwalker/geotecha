@ECHO ******************
@ECHO remove geotecha install stuff
@ECHO by Rohan Walker, November 2013
@ECHO ******************

set folder="C:\Users\Rohan Walker\Documents\GitHub\geotecha\build"

IF EXIST %folder% (
    rmdir %folder% /s /q
)

set folder="C:\Users\Rohan Walker\Documents\GitHub\geotecha\dist"
IF EXIST %folder% (
    rmdir %folder% /s /q
)


set folder="C:\Users\Rohan Walker\Documents\GitHub\geotecha\geotecha.egg-info"
IF EXIST %folder% (
    rmdir %folder% /s /q
)


set folder="C:\Python27\Lib\site-packages\geotecha-0.1.0-py2.7-win32.egg"
IF EXIST %folder% (
    rmdir %folder% /s /q
)
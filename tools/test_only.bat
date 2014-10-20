@REM "@" means don't display the commands that come after the "@".

@ECHO ******************
@ECHO Build and install geotecha python module and run nosetests
@ECHO by Rohan Walker, October 2014
@ECHO ******************
@REM cd out of geotecha folder or nosetests will run on source code rather than installed pakcage (not exactly sure where nosetests searches first.  Drill down on current directory or what?)
cd %systemroot%
nosetests geotecha -v --with-doctest --doctest-options=+ELLIPSIS
@pause
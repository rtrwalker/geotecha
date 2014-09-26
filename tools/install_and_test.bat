@REM "@" means don't display the commands that come after the "@".

@ECHO ******************
@ECHO Install geotecha python module and run nosetests
@ECHO by Rohan Walker, May 2013
@ECHO ******************
cd "C:\Users\Rohan Walker\Documents\GitHub\geotecha"
python setup.py install
python setup.py clean --all
@REM cd out of spec1d folder or nosetests will run on source code rather than installed pakcage (not exactly sure where nosetests searches first.  Drill down on currnet directory or what?)
cd c:\
nosetests geotecha -v --with-doctest --doctest-options=+ELLIPSIS
@pause
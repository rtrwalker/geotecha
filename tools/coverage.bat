@REM "@" means don't display the commands that come after the "@".

@ECHO ******************
@ECHO nosetest with coverage 
@ECHO by Rohan Walker, March 2014
@ECHO ******************
cd "C:\Users\Rohan Walker\Documents\GitHub\geotecha\geotecha"
nosetests --with-coverage --cover-erase --cover-package=geotecha --cover-tests --cover-html --cover-html-dir "C:\Users\Rohan Walker\Documents\cover"
@pause
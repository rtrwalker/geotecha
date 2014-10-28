@REM "@" means don't display the commands that come after the "@".

@ECHO ******************
@ECHO Build and Install geotecha python module then build docs
@ECHO by Rohan Walker, October 2014
@ECHO ******************
@REM assuming run by clicking on bat file in tools directory of source
cd ..
python setup.py build --compiler=mingw32
python setup.py install
python setup.py clean --all
cd docs
make html
@pause

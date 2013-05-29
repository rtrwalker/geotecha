set myfile="log.txt"
@ECHO Will need to press a key to finish.
install_and_test.bat 1>%myfile% 2>&1
@pause
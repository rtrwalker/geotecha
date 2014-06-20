import nose

#import os
#os.chdir('C:\\')
#print(os.path.abspath(os.curdir))
'nose ignores 1st argument http://stackoverflow.com/a/7070571/2530083'
nose.main(argv=['nose_ignores_1st_arg','geotecha', '-v', '--with-doctest'])

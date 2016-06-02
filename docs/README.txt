Note the 'geotecha_examples' 'folder' is a symlink that I created (If not there you must create it!) using:

mklink /J "C:\Users\Rohan Walker\Documents\GitHub\geotecha\docs\geotecha_examples\" "..\examples\"
or
mklink /J "C:\Users\rohanw\Documents\GitHub\geotecha\docs\geotecha_examples\" "..\examples\"
you may have to change the absolute reference(run the command in the docs directory)

This is similar to matplotlib docs which uses a similar approach when refering to mpl_examples

This has to be done because the pyplot extension for sphinx does not like relative references.

References:

Matplotlib use case:
http://matplotlib.org/devel/documenting_mpl.html#figures

Symlink stuff
http://www.howtogeek.com/howto/16226/complete-guide-to-symbolic-links-symlinks-on-windows-or-linux/
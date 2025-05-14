# Global variables
This metric measures the number of global variables in a file. This includes both public and file-local global variables.

Globals in general should be avoided as it is too easy for them to be modified in other parts of the code. It also pollutes the name space and public globals expose internal representation. This is particularly important in multi-threaded applications as every global is likely going to require some synchronization mechanism to ensure that they are accessed and written in the correct order.


## Recommendation
Try to eliminate globals by passing them as parameters to functions. If global state is necessary, try to encapsulate related variables into structs and classes. In C++ it is preferable to group state as well as functions into classes.


## References
* [Global variables](http://www.learncpp.com/cpp-tutorial/42-global-variables/)
* [C++: Global warning](http://www-h.eng.cam.ac.uk/help/tpl/languages/C++/globals.html)
* [Minimize the scope of variables and functions](https://www.securecoding.cert.org/confluence/display/c/DCL19-C.+Minimize+the+scope+of+variables+and+functions)

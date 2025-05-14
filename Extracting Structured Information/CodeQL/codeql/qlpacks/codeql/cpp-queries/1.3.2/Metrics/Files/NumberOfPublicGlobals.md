# Public global variables
This metric measures the number of public global variables in a file.

Global variables can cause several problems:

* lack of encapsulation, because state is exposed to the whole program, rather than a single module
* difficulty maintaining the software, because code using or manipulating the value of the global may be scattered across the software, so its hard to track them all down when making a change
* defects in multi-threading, because the code that uses globals is usually not thread-safe

## Recommendation
Make global variables local to a file, and provide functions for manipulating the value of the variable in a thread-safe way. If possible, make the variable an instance variable in a class or struct that is passed from function to function as necessary so that there is no global state.


## References
* [Global variables](http://www.learncpp.com/cpp-tutorial/42-global-variables/)
* [C++: Global warning](http://www-h.eng.cam.ac.uk/help/tpl/languages/C++/globals.html)
* [Minimize the scope of variables and functions](https://www.securecoding.cert.org/confluence/display/c/DCL19-C.+Minimize+the+scope+of+variables+and+functions)

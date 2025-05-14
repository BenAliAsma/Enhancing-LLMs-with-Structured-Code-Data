# Include header files only
This rule finds `#include` pre-processor directives that include a non-header (\*.h, \*.hpp) file. The exception is header files that define template classes or functions. These header files can include the implementation files, which are then considered to be part of the header file. Including source files can create unnecessarily large object files, and would also expose the entirety of the implementation instead of just its interface (especially in C with its limited set of scoping keywords).

An exception to these rules are files that contain template definitions. These files can include `.c` and `.cpp` files that contain the implementation of the template. No code is actually generated for templates until they are used, and so their definition (not just the declaration) has to be included in any file that uses the template.


## Recommendation
Only include header files. Refactor the code to put class declarations and function prototypes into header files, and their definitions in source files (`*.c, *.cpp`) files.


## Example

```cpp
#include <iostream.h> //correct: iostream.h is a header file
#include "implementation.c" //wrong: include header files only
...


```

## References
* AV Rule 32, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* Scott Meyers. Item 38, Effective C++: 50 Specific Ways to Improve Your Programs and Design, 2nd Edition. Addison-Wesley, 1998.
* [Header files](http://www.learncpp.com/cpp-tutorial/19-header-files/)

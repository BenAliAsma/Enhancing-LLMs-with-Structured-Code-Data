# Usage of macros
This metric measures the percentage of lines of code in each file that use macros.

Macros are useful for defining constants in header files to avoid having magic constants scattered throughout a project. However, files with a high percentage of lines that use macros are likely to be using macros to encapsulate functionality. Such use can lead to code that is hard to read and understand, because standard assumptions about the control-flow of a function, and how many times expressions are evaluated, may no longer hold.

With regards to performance, modern compilers usually perform aggressive inlining and the difference between a macro invocation and a function call is negligible. Functions have much better error checking and IDE support than macros.


```cpp
/* Example of good macros */
#define TCP_NODELAY             1
#define TCP_MAXSEG              2
#define TCP_CORK                3

/*
 * In contrast, functions that use this macro are hard to read without
 * knowing its exact definition
 */
#define JSKW_TEST_GUESS(index)  i = (index); goto test_guess;

```

## Recommendation
Try to limit the use of macros to constants. Complicated uses of macros can often be replaced by functions.


## References
* MSDN Library: [Macros (C/C++)](https://docs.microsoft.com/en-us/cpp/preprocessor/macros-c-cpp)
* [Rejuvenating C++ Programs through Demacrofication](http://www.stroustrup.com/icsm-2012-demacro.pdf)
* [Outgrowing Macrophobia](http://www.idinews.com/macroPhobe.html)

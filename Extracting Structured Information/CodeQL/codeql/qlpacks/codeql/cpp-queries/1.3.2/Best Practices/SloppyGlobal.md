# Short global name
This rule finds global variables which have a name of length three characters or less. It is particularly important to use descriptive names for global variables. Use of a clear naming convention for global variables helps document their use, avoids pollution of the namespace and reduces the risk of shadowing with local variables.


## Recommendation
Review the purpose of the each global variable flagged by this rule and update each name to reflect the purpose of the variable.


## References
* Mats Henricson and Erik Nyquist, *Industrial Strength C++*, published by Prentice Hall PTR (1997). Chapter 1: Naming, Rec 1.1 ([PDF](https://web.archive.org/web/20190919025638/https://mongers.org/industrial-c++/)).
* [Global variables](http://www.learncpp.com/cpp-tutorial/42-global-variables/).

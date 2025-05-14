# Use of goto
Use of `goto` statements makes code more difficult to understand and maintain. Consequently, the use of `goto` statements is deprecated except as a mechanism for breaking out of multiple nested loops, or jumping to error-handling code at the end of a function. This rule identifies complex (and therefore hard to understand) uses of `goto` statements. Function bodies that include multiple `goto` statements that jump forward and multiple `goto` statements that jump backwards are highlighted.


## Recommendation
In most cases the code can be rewritten and/or rearranged by:

* using structured control flow constructs, such as `if`, `while`, and `for`;
* using `break` or `continue` to stop a loop iteration early; or
* using `return` to exit a function early
The `goto` statement may be the best solution for breaking out of deeply nested loops, or for jumping to error handling code, without adversely affecting the readability of the function. Such uses will not be flagged by this rule.


## References
* *Joint Strike Fighter Air Vehicle C++ Coding Standards*, AV Rule 189. Lockheed Martin Corporation, 2005.
* E. W. Dijkstra Archive: [A Case against the GO TO Statement (EWD-215)](http://www.cs.utexas.edu/users/EWD/transcriptions/EWD02xx/EWD215.html).
* MSDN Library: [goto Statement (C++)](https://docs.microsoft.com/en-us/cpp/cpp/goto-statement-cpp).
* Mats Henricson and Erik Nyquist, *Industrial Strength C++*, Rule 4.6. Prentice Hall PTR, 1997. ([PDF](https://web.archive.org/web/20190919025638/https://mongers.org/industrial-c++/)).
* cplusplus.com: [Control Structures](http://www.cplusplus.com/doc/tutorial/control/).

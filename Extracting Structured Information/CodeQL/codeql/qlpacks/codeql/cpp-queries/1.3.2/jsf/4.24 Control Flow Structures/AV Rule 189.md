# AV Rule 189
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

Use of goto statements makes code more difficult to understand and maintain. Consequently, the use of goto statements is deprecated except as a mechanism for breaking out of multiple nested loops. This rule identifies any goto statements that are called directly or from a single nested loop as violations.


## Recommendation
In most cases you should replace the goto statement in the highlighted code with an alternative jump statement (break, continue or return). In deeply nested loops you may need to use a goto statement because the break statement only exits from one level of the loop.


## References
* AV Rule 189, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* [A Case against the GO TO Statement (EWD-215).](http://www.cs.utexas.edu/users/EWD/transcriptions/EWD02xx/EWD215.html)
* MSDN Library: [goto Statement (C++)](https://docs.microsoft.com/en-us/cpp/cpp/goto-statement-cpp).
* Mats Henricson and Erik Nyquist, *Industrial Strength C++*, published by Prentice Hall PTR (1997). Chapter 4: Control Flow, Rule 4.6 ([PDF](https://web.archive.org/web/20190919025638/https://mongers.org/industrial-c++/)).
* [www.cplusplus.com Control Structures](http://www.cplusplus.com/doc/tutorial/control/)

# Magic strings
A *magic string* is a string literal (for example, `"SELECT"`, `"127.0.0.1"`) that is used in the middle of a block of code without explanation. It is considered good practice to avoid magic strings by assigning the strings to named constants and using the named constants instead. The reasons for this are twofold:

1. A string in isolation can be inexplicable to later programmers, whereas a named constant (such as `SMTP_HELO`) is more readily understood.
1. Using the same named constant in many places, makes the code much easier to update if the requirements change (for example, a protocol is updated).
This rule finds magic strings for which there is no pre-existing named constant.


## Recommendation
Consider replacing the magic string with a new named constant.


## References
* [Magic string (Wikipedia)](http://en.wikipedia.org/wiki/Magic_string)
* Mats Henricson and Erik Nyquist, *Industrial Strength C++*, published by Prentice Hall PTR (1997). Chapter 5: Object Life Cycle, Rec 5.4 ([PDF](https://web.archive.org/web/20190919025638/https://mongers.org/industrial-c++/)).
* [DCL06-C. Use meaningful symbolic constants to represent literal values](https://www.securecoding.cert.org/confluence/display/c/DCL06-C.+Use+meaningful+symbolic+constants+to+represent+literal+values)

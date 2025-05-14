# Inconsistent operation on return value
If the same operation (e.g. `free`, `delete`, `close`, etc.) is usually performed on the result of a method call, then any instances where it is not performed may indicate misuse of the API and could cause resource leaks or other issues.


## Recommendation
Examine the code to determine if the return value is treated correctly in this particular case.


## References
* Tutorialspoint - The C++ Programming Language: [C++ Dynamic Memory](http://www.tutorialspoint.com/cplusplus/cpp_dynamic_memory.htm)
* Common Weakness Enumeration: [CWE-252](https://cwe.mitre.org/data/definitions/252.html).

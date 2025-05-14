# Operator Find Incorrectly Used Exceptions
Finding places for the dangerous use of exceptions.


## Example
The following example demonstrates erroneous and fixed methods for using exceptions.


```cpp
...
throw ("my exception!",546); // BAD
...
throw errorFunc("my exception!",546); // GOOD 
...
std::runtime_error("msg error"); // BAD
...
throw std::runtime_error("msg error"); // GOOD
...

```

## References
* CERT CPP Coding Standard: [DCL57-CPP. Do not let exceptions escape from destructors or deallocation functions](https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL57-CPP.+Do+not+let+exceptions+escape+from+destructors+or+deallocation+functions).
* Common Weakness Enumeration: [CWE-703](https://cwe.mitre.org/data/definitions/703.html).
* Common Weakness Enumeration: [CWE-248](https://cwe.mitre.org/data/definitions/248.html).
* Common Weakness Enumeration: [CWE-390](https://cwe.mitre.org/data/definitions/390.html).

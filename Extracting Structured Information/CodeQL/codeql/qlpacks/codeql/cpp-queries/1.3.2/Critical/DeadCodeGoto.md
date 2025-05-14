# Dead code due to goto or break statement
Code immediately following a `goto` or `break` statement will not be executed, unless there is a label or switch case. When the code is necessary, this leads to logical errors or resource leaks. If the code is unnecessary, it may confuse readers.


## Recommendation
If the unreachable code is necessary, move the `goto` or `break` statement to after the code. Otherwise, delete the unreachable code.


## Example

```cpp
goto err1;
free(pointer); // BAD: this line is unreachable
err1: return -1;

free(pointer); // GOOD: this line is reachable
goto err2;
err2: return -1;

```

## References
* The CERT C Secure Coding Standard: [MSC12-C. Detect and remove code that has no effect or is never executed](https://wiki.sei.cmu.edu/confluence/display/c/MSC12-C.+Detect+and+remove+code+that+has+no+effect+or+is+never+executed).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).

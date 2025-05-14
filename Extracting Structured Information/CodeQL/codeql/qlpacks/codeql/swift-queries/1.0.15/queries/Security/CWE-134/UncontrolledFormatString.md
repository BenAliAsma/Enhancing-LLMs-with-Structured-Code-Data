# Uncontrolled format string
Passing untrusted format strings to functions that use `printf` style formatting can lead to buffer overflows and data representation problems. An attacker may be able to exploit this weakness to crash the program or obtain sensitive information from its internal state.


## Recommendation
Use a constant string literal for the format string to prevent the possibility of data flow from an untrusted source. This also helps to prevent errors where the format arguments do not match the format string.

If the format string cannot be constant, ensure that it comes from a secure data source or is compiled into the source code. If you need to include a string value from the user, use an appropriate specifier (such as `%@`) in the format string and include the user provided value as a format argument.


## Example
In this example, the format string includes a user-controlled `inputString`:


```swift

print(String(format: "User input: " + inputString)) // vulnerable

```
To fix it, make `inputString` a format argument rather than part of the format string, as in the following code:


```swift

print(String(format: "User input: %@", inputString)) // fixed

```

## References
* OWASP: [Format string attack](https://owasp.org/www-community/attacks/Format_string_attack).
* Common Weakness Enumeration: [CWE-134](https://cwe.mitre.org/data/definitions/134.html).

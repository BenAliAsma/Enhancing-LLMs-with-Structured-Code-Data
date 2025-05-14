# Too few arguments to formatting function
Each call to the `printf` function, or a related function, should include the number of arguments defined by the format. Passing the function more arguments than required is harmless (although it may be indicative of other defects). However, passing the function fewer arguments than are defined by the format can be a security vulnerability since the function will process the next item on the stack as the missing arguments.

This might lead to an information leak if a sensitive value from the stack is printed. It might cause a crash if a value on the stack is interpreted as a pointer and leads to accessing unmapped memory. Finally, it may lead to a follow-on vulnerability if an attacker can use this problem to cause the output string to be too long or have unexpected contents.


## Recommendation
Review the format and arguments expected by the highlighted function calls. Update either the format or the arguments so that the expected number of arguments are passed to the function.


## Example

```cpp
int main() {
  printf("%d, %s\n", 42); // Will crash or print garbage
  return 0;
}

```

## References
* CERT C Coding Standard: [FIO47-C. Use valid format strings](https://wiki.sei.cmu.edu/confluence/display/c/FIO47-C.+Use+valid+format+strings).
* Microsoft C Runtime Library Reference: [printf, wprintf](https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/printf-printf-l-wprintf-wprintf-l).
* Common Weakness Enumeration: [CWE-234](https://cwe.mitre.org/data/definitions/234.html).
* Common Weakness Enumeration: [CWE-685](https://cwe.mitre.org/data/definitions/685.html).

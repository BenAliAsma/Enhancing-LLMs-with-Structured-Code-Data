# Wrong type of arguments to formatting function
Each call to the `printf` function or a related function should include the type and sequence of arguments defined by the format. If the function is passed arguments of a different type or in a different sequence then the arguments are reinterpreted to fit the type and sequence expected, resulting in unpredictable behavior.


## Recommendation
Review the format and arguments expected by the highlighted function calls. Update either the format or the arguments so that the expected type and sequence of arguments are passed to the function.


## Example
In the following example, the wrong format specifier is given for an integer format argument:


```cpp
int main() {
  printf("%s\n", 42); // BAD: printf will treat 42 as a char*, will most likely segfault
  return 0;
}

```
The corrected version uses `%i` as the format specifier for the integer format argument:


```cpp
int main() {
  printf("%i\n", 42); // GOOD: printf will treat 42 as an int
  return 0;
}

```

## References
* Microsoft Learn: [Format specification syntax: printf and wprintf functions](https://learn.microsoft.com/en-us/cpp/c-runtime-library/format-specification-syntax-printf-and-wprintf-functions?view=msvc-170).
* cplusplus.com:[](https://cplusplus.com/reference/cstdio/printf/)printf
* CERT C Coding Standard: [FIO47-C. Use valid format strings](https://wiki.sei.cmu.edu/confluence/display/c/FIO47-C.+Use+valid+format+strings).
* Common Weakness Enumeration: [CWE-686](https://cwe.mitre.org/data/definitions/686.html).

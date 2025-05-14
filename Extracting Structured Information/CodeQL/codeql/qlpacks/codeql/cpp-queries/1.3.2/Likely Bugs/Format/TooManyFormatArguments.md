# Too many arguments to formatting function
Each call to the `printf` function, or a related function, should include the number of arguments defined by the format. Passing the function more arguments than required is usually harmless from a security perspective but indicates that different behavior was intended.


## Recommendation
Review the format and arguments expected by the highlighted function calls. Update either the format or the arguments so that the expected number of arguments are passed to the function.


## Example

```cpp
void log_connection_attempt(const char *user_name, char char *ip_address) {
  // This does not print `ip_address`.
  fprintf(stderr, "Connection attempted by '%s'\n", user_name, ip_address);
}

```

## References
* CERT C Coding Standard: [FIO47-C. Use valid format strings](https://wiki.sei.cmu.edu/confluence/display/c/FIO47-C.+Use+valid+format+strings).
* Microsoft C Runtime Library Reference: [printf, wprintf](https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/printf-printf-l-wprintf-wprintf-l).

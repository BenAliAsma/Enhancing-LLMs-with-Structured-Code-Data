# Call to `memset` may be deleted
Calling `memset` or `bzero` on a buffer to clear its contents may get optimized away by the compiler if the buffer is not subsequently used. This is not desirable behavior if the buffer contains sensitive data that could somehow be retrieved by an attacker.


## Recommendation
Use `memset_s` (from C11) instead of `memset`, as `memset_s` will not get optimized away. Alternatively use platform-supplied functions such as `SecureZeroMemory` or `bzero_explicit` that make the same guarantee. Passing the `-fno-builtin-memset` option to the GCC/Clang compiler usually also prevents the optimization. Finally, you can use the public-domain `secure_memzero` function (see references below). This function, however, is not guaranteed to work on all platforms and compilers.


## Example
The following program fragment uses `memset` to erase sensitive information after it is no longer needed:


```c
char password[MAX_PASSWORD_LENGTH];
// read and verify password
memset(password, 0, MAX_PASSWORD_LENGTH);

```
Because of dead store elimination, the call to `memset` may be removed by the compiler (since the buffer is not subsequently used), resulting in potentially sensitive data remaining in memory.

The best solution to this problem is to use the `memset_s` function instead of `memset`:


```c
char password[MAX_PASSWORD_LENGTH];
// read and verify password
memset_s(password, MAX_PASSWORD_LENGTH, 0, MAX_PASSWORD_LENGTH);

```

## References
* CERT C Coding Standard: [MSC06-C. Beware of compiler optimizations](https://wiki.sei.cmu.edu/confluence/display/c/MSC06-C.+Beware+of+compiler+optimizations).
* USENIX: The Advanced Computing Systems Association: [Dead Store Elimination (Still) Considered Harmfuls](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-yang.pdf)
* Common Weakness Enumeration: [CWE-14](https://cwe.mitre.org/data/definitions/14.html).

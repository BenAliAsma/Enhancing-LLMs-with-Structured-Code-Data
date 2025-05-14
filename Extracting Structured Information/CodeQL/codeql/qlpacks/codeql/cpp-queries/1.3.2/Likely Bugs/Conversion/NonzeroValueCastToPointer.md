# Non-zero value cast to pointer
This rule finds compile-time constants other than zero (and some common error markers, like `-1` and `0xdeadbeef`) which are (explicitly or implicitly) converted to a pointer type. This is a dangerous practice, since the meaning of the numerical value of pointers is platform dependent.


## Recommendation
Do not assign integer literals (except NULL) to pointers.


## Example

```cpp
void* p = (void*) 42; //Wrong: non-zero constant assigned to pointer. Is unportable 
                      //and will likely lead to a segfault.

void* p2 = NULL; //Correct: NULL (which is 0) assigned to a pointer

```

## References
* Cplusplus.com: [Pointers](http://www.cplusplus.com/doc/tutorial/pointers/).
* The CERT C Secure Coding Standard: [INT36-C. Converting a pointer to integer or integer to pointer](https://www.securecoding.cert.org/confluence/display/c/INT36-C.+Converting+a+pointer+to+integer+or+integer+to+pointer).

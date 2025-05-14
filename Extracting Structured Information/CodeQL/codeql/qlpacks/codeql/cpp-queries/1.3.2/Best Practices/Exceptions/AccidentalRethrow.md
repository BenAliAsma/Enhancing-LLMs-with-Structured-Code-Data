# Accidental rethrow
The C++ `throw` expression can take several forms. One form throws a new exception, whereas the other re-throws the current exception. In the latter case, if there is no current exception, then the program will be terminated. Presence of a re-throw outside of an exception handling context is often caused by the programmer not knowing what kind of exception to throw.


## Recommendation
The `throw` expression should be changed to throw a particular type of exception.


## Example
```cpp

void bad() {
  /* ... */
  if(error_condition)
    throw;
}

void good() {
  /* ... */
  if(error_condition)
    throw std::exception("Something went wrong.");
}

```

## References
* Open Standards: [Standard for Programming Language C++, draft n3337](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3337.pdf) \[except.throw\], clause 9, page 380.

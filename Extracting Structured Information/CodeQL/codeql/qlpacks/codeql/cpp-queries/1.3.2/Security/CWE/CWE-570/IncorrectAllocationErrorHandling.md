# Incorrect allocation-error handling
Different overloads of the `new` operator handle allocation failures in different ways. If `new T` fails for some type `T`, it throws a `std::bad_alloc` exception, but `new(std::nothrow) T` returns a null pointer. If the programmer does not use the corresponding method of error handling, allocation failure may go unhandled and could cause the program to behave in unexpected ways.


## Recommendation
Make sure that exceptions are handled appropriately if `new T` is used. On the other hand, make sure to handle the possibility of null pointers if `new(std::nothrow) T` is used.


## Example

```cpp
// BAD: the allocation will throw an unhandled exception
// instead of returning a null pointer.
void bad1(std::size_t length) noexcept {
  int* dest = new int[length];
  if(!dest) {
    return;
  }
  std::memset(dest, 0, length);
  // ...
}

// BAD: the allocation won't throw an exception, but
// instead return a null pointer.
void bad2(std::size_t length) noexcept {
  try {
    int* dest = new(std::nothrow) int[length];
    std::memset(dest, 0, length);
    // ...
  } catch(std::bad_alloc&) {
    // ...
  }
}

// GOOD: the allocation failure is handled appropriately.
void good1(std::size_t length) noexcept {
  try {
    int* dest = new int[length];
    std::memset(dest, 0, length);
    // ...
  } catch(std::bad_alloc&) {
    // ...
  }
}

// GOOD: the allocation failure is handled appropriately.
void good2(std::size_t length) noexcept {
  int* dest = new(std::nothrow) int[length];
  if(!dest) {
    return;
  }
  std::memset(dest, 0, length);
  // ...
}

```

## References
* CERT C++ Coding Standard: [MEM52-CPP. Detect and handle memory allocation errors](https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM52-CPP.+Detect+and+handle+memory+allocation+errors).
* Common Weakness Enumeration: [CWE-570](https://cwe.mitre.org/data/definitions/570.html).
* Common Weakness Enumeration: [CWE-252](https://cwe.mitre.org/data/definitions/252.html).
* Common Weakness Enumeration: [CWE-755](https://cwe.mitre.org/data/definitions/755.html).

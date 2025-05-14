# Use of expired stack-address
This rule finds uses of pointers that likely point to local variables in expired stack frames. A pointer to a local variable is only valid until the function returns, after which it becomes a dangling pointer.


## Recommendation
1. If it is necessary to take the address of a local variable, then make sure that the address is only stored in memory that does not outlive the local variable. For example, it is safe to store the address in another local variable. Similarly, it is also safe to pass the address of a local variable to another function provided that the other function only uses it locally and does not store it in non-local memory.
1. If it is necessary to store an address which will outlive the current function scope, then it should be allocated on the heap. Care should be taken to make sure that the memory is deallocated when it is no longer needed, particularly when using low-level memory management routines such as `malloc`/`free` or `new`/`delete`. Modern C++ applications often use smart pointers, such as `std::shared_ptr`, to reduce the chance of a memory leak.

## Example

```cpp
static const int* xptr;

void localAddressEscapes() {
  int x = 0;
  xptr = &x;
}

void example1() {
  localAddressEscapes();
  const int* x = xptr; // BAD: This pointer points to expired stack allocated memory.
}

void localAddressDoesNotEscape() {
  int x = 0;
  xptr = &x;
  // ...
  // use `xptr`
  // ...
  xptr = nullptr;
}

void example2() {
  localAddressDoesNotEscape();
  const int* x = xptr; // GOOD: This pointer does not point to expired memory.
}

```

## References
* Wikipedia: [Dangling pointer](https://en.wikipedia.org/wiki/Dangling_pointer).
* Common Weakness Enumeration: [CWE-825](https://cwe.mitre.org/data/definitions/825.html).

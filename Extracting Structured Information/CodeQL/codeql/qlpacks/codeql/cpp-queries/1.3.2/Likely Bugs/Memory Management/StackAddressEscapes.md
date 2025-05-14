# Local variable address stored in non-local memory
This rule finds assignments that might store the address of a local variable in non-local memory. The address of the local variable is only valid until the function returns, after which it becomes a dangling pointer. Such code is safe if the dangling pointer is never used after the function returns, but it is not a recommended coding practice. There is also a risk that the code is not thread-safe, unless the non-local memory is protected by a mutex.


## Recommendation
1. If it is necessary to take the address of a local variable, then make sure that the address is only stored in memory that does not outlive the local variable. For example, it is safe to store the address in another local variable. Similarly, it is also safe to pass the address of a local variable to another function provided that the other function only uses it locally and does not store it in non-local memory.
1. If it is necessary to store an address which will outlive the current function scope, then it should be allocated on the heap. Care should be taken to make sure that the memory is deallocated when it is no longer needed, particularly when using low-level memory management routines such as `malloc`/`free` or `new`/`delete`. Modern C++ applications often use smart pointers, such as `std::shared_ptr`, to reduce the chance of a memory leak.

## Example

```cpp
static const int* xptr;

void example1() {
  int x = 0;
  xptr = &x; // BAD: address of local variable stored in non-local memory.
}

void example2() {
  static const int x = 0;
  xptr = &x; // GOOD: storing address of static variable is safe.
}

```

## References
* Wikipedia: [Dangling pointer](https://en.wikipedia.org/wiki/Dangling_pointer).

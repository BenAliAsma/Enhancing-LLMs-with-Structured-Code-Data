# Return c_str of local std::string
The `c_str` method of `std::string` returns a raw pointer to the memory buffer owned by the `std::string`. The pointer is only safe to use while the `std::string` is still in scope. When the `std::string` goes out of scope, its destructor is called and the memory is deallocated, so it is no longer safe to use the pointer.


## Example

```cpp
#include <string>

const char* hello() {
  std::string str("hello");
  return str.c_str();  // BAD: returning a dangling pointer.
}

```

## Recommendation
Avoid using C-strings. It is much safer to use `std::string` throughout the codebase, because then the memory will be automatically managed.

If C-strings must be used, then be very careful to make sure that there are no pointers to the string that can outlive the lifetime of the string. For example, if the C-string is stack-allocated then it unsafe to store a pointer to the string anywhere on the heap unless you are sure that the heap memory will be deallocated before the end of the function.


## Example

```cpp
#include <string>

std::string hello() {
  std::string str("hello");
  return str;  // GOOD: returning a std::string is safe.
}

```

## References
* cplusplus.com: [std::string::c_str](http://www.cplusplus.com/reference/string/string/c_str/)
* stackoverflow.com: [What is std::string::c_str() lifetime?](https://stackoverflow.com/questions/6456359/what-is-stdstringc-str-lifetime/6456408)

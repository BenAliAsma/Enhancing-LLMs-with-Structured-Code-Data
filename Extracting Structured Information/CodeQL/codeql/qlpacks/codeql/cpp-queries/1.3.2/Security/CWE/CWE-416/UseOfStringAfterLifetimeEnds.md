# Use of string after lifetime ends
Calling `c_str` on a `std::string` object returns a pointer to the underlying character array. When the `std::string` object is destroyed, the pointer returned by `c_str` is no longer valid. If the pointer is used after the `std::string` object is destroyed, then the behavior is undefined.

Typically, this problem occurs when a `std::string` is returned by a function call (or overloaded operator) by value, and the result is not immediately stored in a variable by value or reference in a way that extends the lifetime of the temporary object. The resulting temporary `std::string` object is destroyed at the end of the containing expression statement, along with any memory returned by a call to `c_str`.


## Recommendation
Ensure that the pointer returned by `c_str` does not outlive the underlying `std::string` object.


## Example
The following example concatenates two `std::string` objects, and then converts the resulting string to a C string using `c_str` so that it can be passed to the `work` function. However, the underlying `std::string` object that represents the concatenated string is destroyed as soon as the call to `c_str` returns. This means that `work` is given a pointer to invalid memory.


```cpp
#include <string>
void work(const char*);

// BAD: the concatenated string is deallocated when `c_str` returns. So `work`
// is given a pointer to invalid memory.
void work_with_combined_string_bad(std::string s1, std::string s2) {
  const char* combined_string = (s1 + s2).c_str();
  work(combined_string);
}
```
The following example fixes the above code by ensuring that the pointer returned by the call to `c_str` does not outlive the underlying `std::string` objects. This ensures that the pointer passed to `work` points to valid memory.


```cpp
#include <string>
void work(const char*);

// GOOD: the concatenated string outlives the call to `work`. So the pointer
// obtainted from `c_str` is valid.
void work_with_combined_string_good(std::string s1, std::string s2) {
  auto combined_string = s1 + s2;
  work(combined_string.c_str());
}
```

## References
* [MEM50-CPP. Do not access freed memory](https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM50-CPP.+Do+not+access+freed+memory).
* Microsoft Learn: [Temporary objects](https://learn.microsoft.com/en-us/cpp/cpp/temporary-objects?view=msvc-170).
* cppreference.com: [Lifetime of a temporary](https://en.cppreference.com/w/cpp/language/reference_initialization#Lifetime_of_a_temporary).
* Common Weakness Enumeration: [CWE-416](https://cwe.mitre.org/data/definitions/416.html).
* Common Weakness Enumeration: [CWE-664](https://cwe.mitre.org/data/definitions/664.html).

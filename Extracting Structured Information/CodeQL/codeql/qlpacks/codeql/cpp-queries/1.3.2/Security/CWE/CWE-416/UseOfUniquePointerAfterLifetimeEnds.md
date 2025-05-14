# Use of unique pointer after lifetime ends
Calling `get` on a `std::unique_ptr` object returns a pointer to the underlying allocations. When the `std::unique_ptr` object is destroyed, the pointer returned by `get` is no longer valid. If the pointer is used after the `std::unique_ptr` object is destroyed, then the behavior is undefined.


## Recommendation
Ensure that the pointer returned by `get` does not outlive the underlying `std::unique_ptr` object.


## Example
The following example gets a `std::unique_ptr` object, and then converts the resulting unique pointer to a pointer using `get` so that it can be passed to the `work` function. However, the `std::unique_ptr` object is destroyed as soon as the call to `get` returns. This means that `work` is given a pointer to invalid memory.


```cpp
#include <memory>
std::unique_ptr<T> getUniquePointer();
void work(const T*);

// BAD: the unique pointer is deallocated when `get` returns. So `work`
// is given a pointer to invalid memory.
void work_with_unique_ptr_bad() {
  const T* combined_string = getUniquePointer().get();
  work(combined_string);
}
```
The following example fixes the above code by ensuring that the pointer returned by the call to `get` does not outlive the underlying `std::unique_ptr` objects. This ensures that the pointer passed to `work` points to valid memory.


```cpp
#include <memory>
std::unique_ptr<T> getUniquePointer();
void work(const T*);

// GOOD: the unique pointer outlives the call to `work`. So the pointer
// obtainted from `get` is valid.
void work_with_unique_ptr_good() {
  auto combined_string = getUniquePointer();
  work(combined_string.get());
}
```

## References
* [MEM50-CPP. Do not access freed memory](https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM50-CPP.+Do+not+access+freed+memory).
* Common Weakness Enumeration: [CWE-416](https://cwe.mitre.org/data/definitions/416.html).
* Common Weakness Enumeration: [CWE-664](https://cwe.mitre.org/data/definitions/664.html).

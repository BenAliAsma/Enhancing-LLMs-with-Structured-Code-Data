# Use of object after its lifetime has ended
Using an object after its lifetime has ended results in undefined behavior. When an object's lifetime has ended it relinquishes ownership of its resources and the memory it occupied may be reused for other purposes. If the object is accessed after its lifetime has ended, the program may crash or behave in unexpected ways.


## Recommendation
Ensure that no object is accessed after its lifetime has ended. Use RAII ("Resource Acquisition Is Initialization") to manage the lifetime of objects, and avoid manual memory management, if possible.


## Example
The following two examples demonstrate common lifetime violations when working with the C++ standard library.

The `bad_call_c_api` function contains a use of an expired lifetime. First, a temporary object of type `std::string` is constructed, and a pointer to its internal buffer is stored in a local variable. Once the `c_str()` call returns, the temporary object is destroyed, and the memory pointed to by `p` is freed. Thus, any attempt to dereference `p` inside `c_api` will result in a use-after-free vulnerability. The `good_call_c_api` function contains a fixed version of the first example. The variable `hello` is declared as a local variable, and the pointer to its internal buffer is stored in `p`. The lifetime of hello outlives the call to `c_api`, so the pointer stored in `p` remains valid throughout the call to `c_api`.


```cpp
void c_api(const char*);

void bad_call_c_api() {
  // BAD: the memory returned by `c_str()` is freed when the temporary string is destroyed
  const char* p = std::string("hello").c_str();
  c_api(p);
}

void good_call_c_api() {
  // GOOD: the "hello" string outlives the pointer returned by `c_str()`, so it's safe to pass it to `c_api()`
  std::string hello("hello");
  const char* p = hello.c_str();
  c_api(p);
}

```
The `bad_remove_even_numbers` function demonstrates a potential issue with iterator invalidation. Each C++ standard library container comes with a specification of which operations invalidates iterators pointing into the container. For example, calling `erase` on an object of type `std::vector<T>` invalidates all its iterators, and thus any attempt to dereference the iterator can result in a use-after-free vulnerability. The `good_remove_even_numbers` function contains a fixd version of the third example. The `erase` function returns an iterator to the element following the last element removed, and this return value is used to ensure that `it` remains valid after the call to `erase`.


```cpp
void bad_remove_even_numbers(std::vector<int>& v) {
    // BAD: the iterator is invalidated after the call to `erase`.
  for(std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
    if(*it % 2 == 0) {
      v.erase(it);
    }
  }
}

void good_remove_even_numbers(std::vector<int>& v) {
  // GOOD: `erase` returns the iterator to the next element.
  for(std::vector<int>::iterator it = v.begin(); it != v.end(); ) {
    if(*it % 2 == 0) {
      it = v.erase(it);
    } else {
      ++it;
    }
  }
}
```

## References
* CERT C Coding Standard: [MEM30-C. Do not access freed memory](https://wiki.sei.cmu.edu/confluence/display/c/MEM30-C.+Do+not+access+freed+memory).
* OWASP: [Using freed memory](https://owasp.org/www-community/vulnerabilities/Using_freed_memory).
* [Lifetime safety: Preventing common dangling](https://github.com/isocpp/CppCoreGuidelines/blob/master/docs/Lifetime.pdf)
* [Containers library](https://en.cppreference.com/w/cpp/container)
* [RAII](https://en.cppreference.com/w/cpp/language/raii)
* Common Weakness Enumeration: [CWE-416](https://cwe.mitre.org/data/definitions/416.html).

# Iterator to expired container
Using an iterator owned by a container after the lifetime of the container has expired can lead to undefined behavior. This is because the iterator may be invalidated when the container is destroyed, and dereferencing an invalidated iterator is undefined behavior. These problems can be hard to spot due to C++'s complex rules for temporary object lifetimes and their extensions.


## Recommendation
Never create an iterator to a temporary container when the iterator is expected to be used after the container's lifetime has expired.


## Example


The rules for lifetime extension ensures that the code in `lifetime_of_temp_extended` is well-defined. This is because the lifetime of the temporary container returned by `get_vector` is extended to the end of the loop. However, prior to C++23, the lifetime extension rules do not ensure that the container returned by `get_vector` is extended in `lifetime_of_temp_not_extended`. This is because the temporary container is not bound to a rvalue reference.


```cpp
#include <vector>

std::vector<int> get_vector();

void use(int);

void lifetime_of_temp_extended() {
  for(auto x : get_vector()) {
    use(x); // GOOD: The lifetime of the vector returned by `get_vector()` is extended until the end of the loop.
  }
}

// Writes the the values of `v` to an external log and returns it unchanged.
const std::vector<int>& log_and_return_argument(const std::vector<int>& v);

void lifetime_of_temp_not_extended() {
  for(auto x : log_and_return_argument(get_vector())) {
    use(x); // BAD: The lifetime of the vector returned by `get_vector()` is not extended, and the behavior is undefined.
  }
}

```
To fix `lifetime_of_temp_not_extended`, consider rewriting the code so that the lifetime of the temporary object is extended. In `fixed_lifetime_of_temp_not_extended`, the lifetime of the temporary object has been extended by storing it in an rvalue reference.


```cpp
void fixed_lifetime_of_temp_not_extended() {
  auto&& v = get_vector();
  for(auto x : log_and_return_argument(v)) {
    use(x); // GOOD: The lifetime of the container returned by `get_vector()` has been extended to the lifetime of `v`.
  }
}

```

## References
* CERT C Coding Standard: [MEM30-C. Do not access freed memory](https://wiki.sei.cmu.edu/confluence/display/c/MEM30-C.+Do+not+access+freed+memory).
* OWASP: [Using freed memory](https://owasp.org/www-community/vulnerabilities/Using_freed_memory).
* [Lifetime safety: Preventing common dangling](https://github.com/isocpp/CppCoreGuidelines/blob/master/docs/Lifetime.pdf)
* [Containers library](https://en.cppreference.com/w/cpp/container)
* [Range-based for loop (since C++11)](https://en.cppreference.com/w/cpp/language/range-for)
* Common Weakness Enumeration: [CWE-416](https://cwe.mitre.org/data/definitions/416.html).
* Common Weakness Enumeration: [CWE-664](https://cwe.mitre.org/data/definitions/664.html).

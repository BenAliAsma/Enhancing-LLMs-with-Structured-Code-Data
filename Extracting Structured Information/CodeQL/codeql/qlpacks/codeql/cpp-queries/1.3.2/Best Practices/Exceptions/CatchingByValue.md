# Catching by value
Catching an exception by value will create a new local variable which is a copy of the originally thrown object. Creating the copy is slightly wasteful, but not catastrophic. More worrisome is the fact that if the type being caught is a strict supertype of the originally thrown type, then the copy might not contain as much information as the original exception.


## Recommendation
The parameter to the `catch` block should have its type changed from `T` to `T&` or `const T&`.


## Example
```cpp

void bad() {
  try {
    /* ... */
  }
  catch(std::exception a_copy_of_the_thrown_exception) {
    // Do something with a_copy_of_the_thrown_exception
  }
}

void good() {
  try {
    /* ... */
  }
  catch(const std::exception& the_thrown_exception) {
    // Do something with the_thrown_exception
  }
}

```

## References
* C++ FAQ: [ What should I throw?](https://isocpp.org/wiki/faq/exceptions#what-to-throw), [ What should I catch?](https://isocpp.org/wiki/faq/exceptions#what-to-catch).
* Wikibooks: [ Throwing objects](http://en.wikibooks.org/wiki/C%2B%2B_Programming/Exception_Handling#Throwing_objects).

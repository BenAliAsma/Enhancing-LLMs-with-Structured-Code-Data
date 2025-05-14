# Throwing pointers
As C++ is not a garbage collected language, exceptions should not be dynamically allocated. Dynamically allocating an exception puts an onus on every `catch` site to ensure that the memory is freed.

As a special case, it is permissible to throw anything derived from Microsoft MFC's `CException` class as a pointer. This is for historical reasons; modern code and modern frameworks should not throw pointer values.


## Recommendation
The `new` keyword immediately following the `throw` keyword should be removed. Any `catch` sites which previously caught the pointer should be changed to catch by reference or `const` reference.


## Example
```cpp

void bad() {
  throw new std::exception("This is how not to throw an exception");
}

void good() {
  throw std::exception("This is how to throw an exception");
}

```

## References
* C++ FAQ: [ What should I throw?](https://isocpp.org/wiki/faq/exceptions#what-to-throw), [ What should I catch?](https://isocpp.org/wiki/faq/exceptions#what-to-catch).
* Wikibooks: [ Throwing objects](http://en.wikibooks.org/wiki/C%2B%2B_Programming/Exception_Handling#Throwing_objects).

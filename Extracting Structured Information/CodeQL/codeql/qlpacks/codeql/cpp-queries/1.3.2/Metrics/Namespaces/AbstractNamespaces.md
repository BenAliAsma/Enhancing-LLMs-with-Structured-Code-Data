# Abstract namespaces
This query finds namespaces that have an abstractness greater than 0.20.

Abstractness measures the proportion of abstract types in a package relative to the total number of types in that package. A metric value close to 1 indicates a highly abstract package that is also unstable. The class hierarchy is probably over-engineered, and the abstract types are unlikely to be much used.


## Recommendation
Consider reducing the level of abstraction by simplifying the class hierarchy.


## References
* C++ reference: [Namespaces](https://en.cppreference.com/w/cpp/language/namespace)
* Geeks for Geeks: [Abstraction in C++](https://www.geeksforgeeks.org/abstraction-in-c/)

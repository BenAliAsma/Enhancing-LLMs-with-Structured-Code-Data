# Namespaces far from main line
This query finds namespaces that do not have a good balance between abstractness and stability where:

* Abstractness measures the proportion of abstract types in a namespace relative to the total number of types in that namespace.
* Instability measures the level of expectation that changes to other namespaces will affect this namespace.
The metric tries to capture the trade-off between abstractness and instability. For an ideal balance, the sum of abstractness and instability should be one. That is, a package is completely abstract and stable (abstractness=1 and instability=0) or it is concrete and instable (abstractness=0 and instability=1). This query measures the distance between the balance in the code base and the ideal.


## References
* C++ reference: [Namespaces](https://en.cppreference.com/w/cpp/language/namespace)
* Geeks for Geeks: [Abstraction in C++](https://www.geeksforgeeks.org/abstraction-in-c/)

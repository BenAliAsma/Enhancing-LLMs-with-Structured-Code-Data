# Rule of three
This query finds classes that define a destructor, a copy constructor, or a copy assignment operator, but not all three of them. The compiler generates default implementations for these functions, and since they deal with similar concerns it is likely that if the default implementation of one of them is not satisfactory, then neither are those of the others.

The query flags any such class with a warning, and also display the list of generated warnings in the result view.


## Recommendation
Explicitly define the missing functions.


## References
* Wikipedia: [Rule of three (C++ programming)](http://en.wikipedia.org/wiki/Rule_of_three_(C%2B%2B_programming))

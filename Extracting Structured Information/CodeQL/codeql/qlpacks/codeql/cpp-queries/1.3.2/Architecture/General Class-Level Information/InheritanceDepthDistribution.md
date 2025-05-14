# Inheritance depth distribution
This query shows the distribution of inheritance depth across all types, that is, classes. Library types are ignored.

The result of this query is a line graph showing, for each number *n*, how many types have an inheritance depth of *n*, where the inheritance depth of a type is the length of a longest path in the inheritance hierarchy from top class to the type.


## Recommendation
The depth of a type is an indication of how deeply nested a type is in a given design. Very deep types can be an indication of over-engineering, whereas a system with predominantly shallow types may not be exploiting object-orientation to the full.


## References
* Shyam R. Chidamber and Chris F. Kemerer, *[A Metrics Suite for Object Oriented Design ](http://www.pitt.edu/~ckemerer/CK%20research%20papers/MetricForOOD_ChidamberKemerer94.pdf)*. IEEE Transactions on Software Engineering, 20(6), pages 476-493, June 1994.
* Lutz Prechelt, Barbara Unger, Michael Philippsen, and Walter Tich, *[A Controlled Experiment on Inheritance Depth as a Cost Factor for Code Maintenance ](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.159.2229&rep=rep1&type=pdf)*. Journal of Systems and Software, 65 (2):115-126, 2003.

# Size of type APIs
This metric counts the number of items in the public API of a reference type. This includes methods, properties, operators, indexers and events. A large public API can be difficult to use.


## Recommendation
Consider whether the type is trying to perform too many functions. If it is then the type should be split up into multiple separate types for each function it performs.


## References
* BlackWasp. [Single Responsibility Principle](http://www.blackwasp.co.uk/SRP.aspx).

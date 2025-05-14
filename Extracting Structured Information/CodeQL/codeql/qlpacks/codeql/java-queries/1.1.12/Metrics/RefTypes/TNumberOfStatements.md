# Number of statements in types
This metric measures the number of statements that occur in a type.

If there are too many statements in a type, it is generally for one of two reasons:

* One or more individual methods of the type contain too many statements, making them hard to understand, difficult to check and a common source of defects (particularly towards the end of the methods, since few people ever read that far). These methods typically lack cohesion because they are trying to do too many things.
* The type contains too many methods, which generally indicates that it is trying to do too much, either at the interface or implementation level or both. It can be difficult for readers to understand because there is a confusing list of operations.

## Recommendation
As described above, types reported as violations by this rule contain one or more methods with too many statements, or the type itself contains too many methods.

* Individual methods of the type that contain too many statements should be refactored into multiple, smaller methods. As a rough guide, methods should be able to fit on a single screen or side of A4. Anything longer than that increases the risk of introducing new defects during routine code changes.
* Types that contain too many methods often lack cohesion and are prime candidates for refactoring.

## References
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.

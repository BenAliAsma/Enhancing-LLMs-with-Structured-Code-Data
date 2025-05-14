# Non-final immutable field
A field of immutable type that is not declared `final`, but is assigned to only in a constructor or static initializer of its declaring type, may lead to defects and makes code less readable. This is because other parts of the code may be based on the assumption that the field has a constant value, and a later modification, which includes an assignment to the field, may invalidate this assumption.


## Recommendation
If a field of immutable type is assigned to only during class or instance initialization, you should usually declare it `final`. This forces the compiler to verify that the field value cannot be changed subsequently, which can help to avoid defects and increase code readability.


## References
* Java Language Specification: [4.12.4 final Variables](https://docs.oracle.com/javase/specs/jls/se11/html/jls-4.html#jls-4.12.4), [8.3.1.2 final Fields](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.3.1.2).

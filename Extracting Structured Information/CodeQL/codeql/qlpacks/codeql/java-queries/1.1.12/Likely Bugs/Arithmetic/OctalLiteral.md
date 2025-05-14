# Use of octal values
An integer literal consisting of a leading `0` digit followed by one or more digits in the range `0-7` is an octal literal. This can lead to two problems:

* An octal literal can be misread by a programmer as a decimal literal.
* A programmer might accidentally start a decimal literal with a zero, so that the compiler treats the decimal literal as an octal literal. For example, `010` is equal to `8`, not `10`.

## Recommendation
To avoid these problems:

* Avoid using octal literals so that programmers do not confuse them with decimal literals. However, if you need to use octal literals, you should add a comment to each octal literal indicating the intention to use octal literals.
* When typing decimal literals, be careful not to begin them with a zero accidentally.

## References
* J. Bloch and N. Gafter, *Java Puzzlers: Traps, Pitfalls, and Corner Cases*, Puzzle 59. Addison-Wesley, 2005.
* Java Language Specification: [Integer Literals](https://docs.oracle.com/javase/specs/jls/se11/html/jls-3.html#jls-3.10.1).

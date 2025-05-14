# Nesting depth
This metric measures the maximum nesting depth in a file. This includes nested branch and loop statements, but not blocks.

Deep nesting makes code very difficult to read and modify, and is also a sign that a function/class has lost cohesion (i.e. is doing too many unrelated things). Try to keep nesting depth below 7 levels.


## Recommendation
Reduce the nesting in the file by putting the code in the inner loops/branches in separate functions.


## References
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)

# Outgoing dependencies per file
This metric measures the number of files that a particular file depends on. A file depends on another file if:

* It calls a function in that file
* It reads or writes a variable declared in that file
* It uses a type declared in that file
The number of other files that a source-file depends on is a way of judging how cohesive it is. A source-file with a single well-defined purpose is likely to have fewer dependencies than a source-file where functions have been "thrown in" over time. The former is desirable, because files with a single well-defined purpose are more easily comprehended and maintained.


## Recommendation
Consider, breaking the file up into smaller ones. This could either be done by splitting up unrelated functionality, or extracting common operations into a library.


## References
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)

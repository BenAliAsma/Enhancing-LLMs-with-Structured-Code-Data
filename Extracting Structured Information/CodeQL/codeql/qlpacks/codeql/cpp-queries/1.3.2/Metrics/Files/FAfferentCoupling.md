# Incoming dependencies per file
This metric measures, for each file, how many other files depend on it.

A file depends on another file if:

* It calls a function in that file
* It reads or writes a variable declared in that file
* It uses a type declared in that file
Many incoming dependencies can be good or bad, depending on the nature of the file.

A large number of incoming dependencies is good for:

* files containing logging functions
* files containing system-wide utility functions
A large number of incoming dependencies may be a problem for:

* files that should be internal to a particular component
* files containing functions or classes with unstable interfaces
A file with many dependencies on it is risky to change, since large parts of a system may be affected. Such files should be well documented and have clean APIs.


## Recommendation
Group widely used utility functions together. Replace calls to a component's internals with uses of its public API (augmenting it if necessary).

Ensure that the public API is well-documented.


## References
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)

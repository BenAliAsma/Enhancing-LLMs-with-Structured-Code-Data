# Classes with too many source dependencies
This rule finds classes that depend on many other types. Depending on too many other classes makes a class vulnerable to changes and defects in its dependencies, and is also a sign that the class lacks cohesion (i.e. a single purpose). These classes could be refactored into smaller, more cohesive classes.


## Recommendation
These classes could probably be refactored into smaller classes with fewer dependencies.


## Example

```cpp
// an include declaration just adds one source dependency, it does not automatically
// add a dependency from this file to all the declarations in stdio.h
#include <stdio.h>
#include <myfile.h> // contains non-static global myfile_err

extern int myfile_err; // this external declaration adds a dependency on myfile.h

class C {
public:
	C() {
		// one dependency for printf:
		printf("Hello world!");
		// one dependency for FILE type, and one for NULL macro:
		FILE fp = NULL;
	}
};


```

## References
* W. Stevens, G. Myers, L. Constantine, *Structured Design*, IBM Systems Journal, 13 (2), 115-139, 1974.
* Microsoft Patterns &amp; Practices Team. [Architectural Patterns and Styles](http://msdn.microsoft.com/en-us/library/ee658117.aspx) *Microsoft Application Architecture Guide, 2nd Edition.* Microsoft Press, 2009.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)

# Undocumented API function
Functions that are called from lots of different places are usually important, and justify having documentation written for them. In particular, if a function is defined in a file, and is called from at least two other files, then the function should probably be documented.

As an exception, because their purpose is usually obvious, it is not necessary to document constructors, destructors, implementations of `operator=`, or functions with fewer than five lines of code.


## Recommendation
Add comments to document the purpose of the function. In particular, ensure that the public API of the function is carefully documented. This reduces the chance that a future change to the function will introduce a defect by changing the API and breaking the expectations of the calling functions.


## References
* C++ Programming Wikibook: [Comments](http://en.wikibooks.org/wiki/C%2B%2B_Programming/Programming_Languages/C%2B%2B/Code/Style_Conventions#Comments)
* Wikipedia: [Need for comments](http://en.wikipedia.org/wiki/Comment_%28computer_programming%29#Need_for_comments)

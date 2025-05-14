# Complex functions
This rule finds functions that make too many calls. Having too many dependencies makes a function vulnerable to changes and defects in those dependencies, and is also a sign that the function lacks cohesion (i.e. lacks a single purpose). These functions can be split into smaller, more cohesive functions.


## Recommendation
Splitting these functions would increase maintainability and readability.


## References
* W. Stevens, G. Myers, L. Constantine, *Structured Design*, IBM Systems Journal, 13 (2), 115-139, 1974.

# Cyclomatic Complexity
This rule computes cyclomatic complexity per function. Cyclomatic complexity is a measure of how complex a function is, and is derived from the number of branching statements in the function. Complex functions can be difficult to understand and maintain, and usually can be split into smaller, less complex functions.

The rule finds functions with high (e.g. &gt;50) cyclomatic complexity.


## Recommendation
With increasing cyclomatic complexity there need to be more test cases that are necessary to achieve a complete branch coverage when testing the function. Try to reduce the function's complexity by splitting it into several more cohesive functions. A particularly effective way of reducing complexity is to put code that handles a particular case in an large `if` or `switch` statement into a separate function with a descriptive name. This not only reduces the complexity of the function, but makes it considerably more readable as the function's descriptive name gives an idea of its purpose without the developer analyzing each line of code.


## References
* Wikipedia: [Cyclomatic complexity](https://en.wikipedia.org/wiki/Cyclomatic_complexity).
* T. McCabe. *A Complexity Measure*. ICSE '76: Proceedings of the 2nd international conference on Software engineering, 1976.

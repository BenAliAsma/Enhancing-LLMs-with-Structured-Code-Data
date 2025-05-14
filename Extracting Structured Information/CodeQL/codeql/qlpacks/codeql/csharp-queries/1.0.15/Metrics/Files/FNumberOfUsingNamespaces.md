# Number of using namespace directives
This metric calculates the number of namespaces that are imported with the `using` directive. Files that have a large number of using directives can be brittle as they may have to be changed if any of their dependencies are changed. It is not uncommon that a file with a high number of outgoing dependencies lacks cohesion because small parts of it rely on small parts of other classes.


## Recommendation
It might be possible to improve the structure of the program by splitting files with a high number of `using` directives into multiple files and classes.


## References
* Robert Cecil Martin. *Agile Software Development: Principles, Patterns and Practices*. Pearson Education. 2002.

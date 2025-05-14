# Cyclomatic complexity of functions
The cyclomatic complexity of a method (or constructor) is the number of possible linearly-independent execution paths through that method (see \[Wikipedia\]). It was originally introduced as a complexity measure by Thomas McCabe \[McCabe\].

A method with high cyclomatic complexity is typically difficult to understand and test.


## Example

```java
int f(int i, int j) {
    int result;
    if(i % 2 == 0) {
        result = i + j;
    }
    else {
        if(j % 2 == 0) {
            result = i * j;
        }
        else {
            result = i - j;
        }
    }
    return result;
}
```
The control flow graph for this method is as follows:

![Control Flow Diagram](./CCyclomaticComplexity_ControlFlow.png)As you can see from the graph, the number of linearly-independent execution paths through the method is 3. Therefore, the cyclomatic complexity is 3.


## Recommendation
Simplify methods that have a high cyclomatic complexity. For example, tidy up complex logic, and/or split methods into multiple smaller methods using the 'Extract Method' refactoring from \[Fowler\].


## References
* M. Fowler, *Refactoring*. Addison-Wesley, 1999.
* T. J. McCabe, *A Complexity Measure*. IEEE Transactions on Software Engineering, SE-2(4), December 1976.
* Wikipedia: [Cyclomatic complexity](http://en.wikipedia.org/wiki/Cyclomatic_complexity).

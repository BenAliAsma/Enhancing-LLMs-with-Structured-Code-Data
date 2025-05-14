# Continue statement that does not continue
A `continue` statement only re-runs the loop if the loop condition is true. Therefore using `continue` in a loop with a constant false condition will never cause the loop body to be re-run, which is misleading.


## Recommendation
Replace the `continue` statement with a `break` statement if the intent is to break from the loop.


## References
* Java Language Specification: [14.13 The do Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.13).

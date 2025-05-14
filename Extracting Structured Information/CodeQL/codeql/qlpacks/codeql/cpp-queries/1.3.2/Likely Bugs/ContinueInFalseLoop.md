# Continue statement that does not continue
A `continue` statement only re-runs the loop if the loop condition is true. Therefore using `continue` in a loop with a constant false condition will never cause the loop body to be re-run, which is misleading.


## Recommendation
Replace the `continue` statement with a `break` statement if the intent is to break from the loop.


## References
* Tutorialspoint - C Programming Language: [Continue Statement in C](https://www.tutorialspoint.com/cprogramming/c_continue_statement.htm).

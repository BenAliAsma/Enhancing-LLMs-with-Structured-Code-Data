# File is not always closed
If a file is opened then it should always be closed again, even if an exception is raised. Failing to ensure that all files are closed may result in failure due to too many open files.


## Recommendation
Ensure that if you open a file it is always closed on exiting the method. Wrap the code between the `open()` and `close()` functions in a `with` statement or use a `try...finally` statement. Using a `with` statement is preferred as it is shorter and more readable.


## Example
The following code shows examples of different ways of closing a file. In the first example, the file is closed only if the method is exited successfully. In the other examples, the file is always closed on exiting the method.


```python
f = open("filename")
    ... # Actions to perform on file
f.close()
# File only closed if actions are completed successfully

with open("filename") as f:
    ...# Actions to perform on file
# File always closed

f = open("filename")
try:
    ... # Actions to perform on file
finally:
    f.close()
# File always closed

```

## References
* Python Language Reference: [The with statement](http://docs.python.org/reference/compound_stmts.html#the-with-statement), [The try statement](http://docs.python.org/reference/compound_stmts.html#the-try-statement).
* Python PEP 343: [The "with" Statement](http://www.python.org/dev/peps/pep-0343).
* Common Weakness Enumeration: [CWE-772](https://cwe.mitre.org/data/definitions/772.html).

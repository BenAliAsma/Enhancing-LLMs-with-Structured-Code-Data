# Leaky catch
Modern C++ code and frameworks should not throw or catch pointers. Older frameworks, such as Microsoft's MFC, do throw and catch pointers. Said pointers will generally point to an exception object allocated on the heap, and therefore need to be freed when they are caught. Failure to free them will result in a memory leak.


## Recommendation
The `catch` block should be augmented to delete the exception pointer.


## Example
```cpp

void bad() {
  try {
    /* ... */
  }
  catch(CException* e) {
    e->ReportError();
  }
}

void good() {
  try {
    /* ... */
  }
  catch(CException* e) {
    e->ReportError();
    e->Delete();
  }
}

```

## References
* MSDN Library for MFC: [Exceptions: Catching and Deleting Exceptions](https://docs.microsoft.com/en-us/cpp/mfc/exceptions-catching-and-deleting-exceptions).
* Common Weakness Enumeration: [CWE-401](https://cwe.mitre.org/data/definitions/401.html).

# 'new' object freed with 'delete[]'
This rule finds `delete[]` expressions that are using a pointer that points to memory allocated using the `new` operator. Behavior in such cases is undefined and should be avoided.

The `new` operator allocates memory for just *one* object, then calls that object's constructor, and `delete` does the opposite. The array `delete[]` operator, however, expects the pointer to be pointing to the first element of an array (which could have header data specifying the length of the array) and would attempt to call the destructor on each element of the 'array', which would likely lead to a segfault due to the invalid header data.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the values of pointers without running the program with all input data.

## Recommendation
Use the `delete` operator when freeing memory allocated with `new`.


## Example

```cpp
Record *ptr = new Record(...);

...

delete [] ptr; // ptr was created using 'new', but was freed using 'delete[]'

```

## References
* S. Meyers. *Effective C++ 3d ed.* pp 73-75. Addison-Wesley Professional, 2005.

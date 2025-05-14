# 'new[]' array freed with 'delete'
This rule finds `delete` expressions that are using a pointer that points to memory allocated using the `new[]` operator. This should be avoided since it results in undefined behavior, as per &sect;5.3.5 of the ISO/IEC C++ Standard:

> "In the first alternative (*delete object*), the value of the operand of `delete` may be a null pointer value, a pointer to a non-array object created by a previous *new-expression*, or a pointer to a sub-object representing a base class of such an object. If not, the behavior is undefined."

Besides being formally undefined, there are two practical reasons why this is likely to cause defects. For the first of these, consider what happens when invoking `X *p = new X[23]`:

1. Sufficient memory is allocated to hold `23` instances of type `X` by invoking `::operator new(sizeof(X) * 23)`.
1. Each of the `23` instances of `X` is constructed at its correct place in memory (as if doing a *placement new*).
If `delete[] p` is subsequently executed, the reverse happens:

1. The destructor for each of the `23` instances of `X` is invoked (as if doing an explicit `(p + i)->~X()`).
1. The memory allocated by `::operator new` is deallocated by invoking `::operator delete(p)`.
By contrast, `delete p` (without the `[]` brackets) would generally assume that `p` points to exactly one instance of `X`, and only call the destructor for that (although this behavior cannot be relied upon, since the results are formally undefined). The practical result of this is that the destructors for the remaining `X` instances, which might do crucial things such as freeing resources, will not be called.

There is also a second practical reason why this may cause a defect. In order to call the destructors of the array elements when `delete[]` is called, the implementation must know the size of array to which `p` points at deletion time. Bearing in mind that `p` is a pointer, and carries no array size information in its type, this information would not in general be available unless the implementation somehow stores it when `new[]` is invoked. There are two common ways in which this is done:

* The most common approach is to allocate a small amount of extra memory (a *header*) before the start of the array and store the size in it. When invoking `delete[] p`, the implementation then just needs to walk back a fixed amount from the passed-in pointer to read the size. The implication of this is that `p` itself (the pointer to the first element in the array) is *not* the pointer returned by `::operator new`, and so it is not safe to call `::operator delete` on it. Instead, it should be called on a pointer that points to the start of the header. Invoking `delete p` would use the wrong address, with potentially catastrophic results.
* An alternative, less common, approach is to store a map from pointers to the sizes of the arrays (if any) to which they point. When invoking `delete[] p`, the implementation looks up the pointer in the map, invokes the relevant number of destructors, deallocates the memory *and removes the pointer from the map*. If `delete p` is called instead, not only will the relevant number of destructors likely not be called (as previously noted), but the pointer will also likely not be removed from the map. In practical terms, this is potentially less of a serious issue than that posed by the first approach, but it should still be avoided.
> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the values of pointers without running the program with all input data.

## Recommendation
Use the `delete[]` operator when freeing memory allocated with `new[]`.


## Example

```cpp
Record* record = new Record[SIZE];

...

delete record; //record was created using 'new[]', but was freed using 'delete'

```

## References
* S. Meyers. *Effective C++ 3d ed.* pp 73-75. Addison-Wesley Professional, 2005.
* *ISO/IEC 14882:2011, Information technology - Programming languages - C++* &sect;5.3.5. International Organization for Standardization, Geneva, Switzerland, 2011.

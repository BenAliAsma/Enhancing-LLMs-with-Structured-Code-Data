# Suspicious call to memset
This rule finds calls to the standard library function `memset` where the third argument (specifying the number of bytes to set) does not appear to correspond with the first argument (the buffer to be written).

Usually, `memset` is employed to initialize dynamically allocated arrays or structs. Since `memset` treats its first argument simply as an array of bytes, the third argument has to specify the size of the buffer *in bytes*. For an array, the size is the number of elements of the array multiplied by the size of one of its elements; for a `struct`, the size is just the size of the struct type.

If `memset` is invoked with a third argument that is not constant and looks like neither of these two cases, there might be a mistake. This could cause a buffer overflow which could in turn cause a segfault or corrupt the contents of other variables in memory.


## Recommendation
For structs the preferred way of computing the size is to use sizeof with the type as the argument. Dereferencing a pointer works but is more prone to mistakes. For arrays the best solution is to take the size of the array rather than the type, since the risk of forgetting to multiply with the number of elements is eliminated. Do not use memset to assign to scalars or pointers when a simple assignment would do.


## Example

```cpp

// correct, memset uses sizeof(type)
int is[10];
memset(is, 0, sizeof(is));
struct S s;
memset(&s, 0, sizeof(struct S));

// incorrect examples
struct T *t1 = (struct T*)malloc(sizeof(struct T));
struct T *t2 = (struct T*)malloc(sizeof(struct T));
// the size of the struct is probably intended
// but this takes the size of a pointer
memset(t2, 0, sizeof(t2));

// correct but discouraged, use sizeof(struct T) instead
memset(t1, 0, sizeof(*t2));
// correct, but it is preferred to do a direct assignment, i.e., t = 0;
memset(&t2, 0, sizeof(t2));

```

## References
* Cplusplus.comn: [memset](http://www.cplusplus.com/reference/clibrary/cstring/memset/)
* MSDN Library: [memset, wmemset](https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/memset-wmemset), [sizeof Operator](https://docs.microsoft.com/en-us/cpp/cpp/sizeof-operator)
* Common Weakness Enumeration: [CWE-676](https://cwe.mitre.org/data/definitions/676.html).

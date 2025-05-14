# Large object passed by value
This rule finds parameters greater than (for example) 64 bytes in size that are passed by value to function calls. It is not good practice to put large objects on the stack, as widespread use of this anti-pattern throughout a software project can easily lead to stack overflows. Performance-wise, the initial copy to the stack is more likely than not to be more expensive than any cost due to indirection caused accessing the object through a pointer. In terms of security, an overrun on an array in an object copied to the stack is a much greater risk than one that happens on the heap.

In C++, there is the added cost of calling the constructor and destructor of the object being passed as well as those of any other objects that it (recursively) contains.


## Recommendation
Pass the address of the object instead. There is usually no good reason for putting anything large on the stack.


## Example

```cpp
typedef struct Names {
    char first[100];
    char last[100];
} Names;

int doFoo(Names n) { //wrong: n is passed by value (meaning the entire structure
                     //is copied onto the stack, instead of just a pointer)
    ...
}

int doBar(const Names &n) { //better, only a reference is passed
    ...
}

```

## References
* S. Meyers. *Effective C++ 3d ed.* pp 86-90. Addison-Wesley Professional, 2005.

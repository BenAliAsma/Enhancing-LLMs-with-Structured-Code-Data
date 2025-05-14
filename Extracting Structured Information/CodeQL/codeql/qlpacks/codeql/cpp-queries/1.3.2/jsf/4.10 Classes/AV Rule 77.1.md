# Constructor with default arguments will be used as a copy constructor
Multi-parameter constructors with default arguments can be signature-compatible with a copy constructor when their default arguments are taken into account. An example would be a constructor for `X` of the form `X(const X& rhs, int i = 0)`. A compiler will use such a constructor as a copy constructor in preference to the default member-wise copy constructor that it would otherwise generate. Since this is usually not what was intended, constructors of the form often do not provide the right semantics for copying objects of the class, making them potentially dangerous.


## Recommendation
Do not declare constructors with default arguments that are signature-compatible with a copy constructor when their default arguments are taken into account.


## Example

```cpp
struct X {
    //This struct will have a compiler-generated copy constructor
    X(const X&, int);
    ...
};

//However, if this is declared later, it will override the compiler-generated
//constructor
X::X(const X& x, int i =0) {
    this-> i = i; //uses the i parameter, instead of x.i
}

C c(1);
C cCopy = c; //would take i to be 0, instead of just copying c(1)

```

## References
* AV Rule 77.1, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.

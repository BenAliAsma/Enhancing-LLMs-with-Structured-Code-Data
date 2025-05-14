# Undisciplined multiple inheritance
This rule ensures that classes conform to a restricted form of multiple inheritance. The restricted form allows inheriting from any number of *interfaces* (i.e. classes which contain only pure virtual functions and only essential data), any number of private implementations, and only one protected implementation. Allowing only one protected base class ensures that the class is only allowed to override functions from that base class (and not accidentally override functions from other base classes). Allowing multiple private implementations allow reuse (but not overriding) of functions from multiple classes.

This restricted form of inheritance is almost identical to Java, where a class may extend (public/protected derive) only a single class, but may implement (public derive) multiple interfaces (pure virtual classes), with the addition of mixin-like behavior provided by the reuse of members by privately deriving from implementation classes \[Bracha and Cook\]. This enforces a public class hierarchy that is predictable and easy to comprehend while giving flexibility and opportunities for code reuse for the non-public implementation.

The indicated class violates the restricted form of multiple inheritance.


## Recommendation
Change the class to conform to the restricted form of multiple inheritance.


## Example

```cpp
//correct:
//  only one protected base,
//  multiple interfaces ("pure virtual" classes), and
//  multiple private implementations
class C : protected Superclass,
          public InterfaceA, public InterfaceB,
          private ImplementationA, private ImplementationB
{
    //implementation
};

//wrong: multiple protected bases
class D : protected Superclass1, protected Superclass2,
          public Interface, private Implementation
{
    //implementation
};


```

## References
* AV Rule 88, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* G. Bracha and W. Cook. *Mixin-based inheritance*. OOPSLA/ECOOP '90: Proceedings of the European conference on object-oriented programming on Object-oriented programming systems, languages, and applications, pp 303-311. 1990.
* S. Meyers. *Effective C++ 3d ed.* pp 192-198. Addison-Wesley Professional, 2005.

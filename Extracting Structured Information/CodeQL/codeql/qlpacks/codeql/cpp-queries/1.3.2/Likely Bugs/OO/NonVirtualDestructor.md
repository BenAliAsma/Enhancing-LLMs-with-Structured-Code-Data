# Non-virtual destructor
This rule finds classes that define a non-virtual destructor, yet they have derived classes that also define a non-virtual destructor. This can prevent proper cleanup of resources as only the destructor of the type of the variable will be called (instead of the type of the object instance).


## Recommendation
Make the destructor virtual.


## Example

```cpp
class Base {
public:
    Resource *p;
    Base() {
        p = createResource();
    }
    //...
    ~Base() {
        //wrong: this destructor is non-virtual, but Base has a derived class
        //with a non-virtual destructor
        freeResource(p);
    }
};

class Derived: public Base {
public:
    Resource *dp;
    Derived() {
        dp = createResource2();
    }
    ~Derived() {
        freeResource2(dp);
    }
};

int f() {
    Base *b = new Derived(); //creates resources for both Base::p and Derived::dp
    //...
    delete b; //will only call Base::~Base(), leaking the resource dp.
        // Change both destructors to virtual to ensure they are both called.
}

```

## References
* R. Chen, [When should your destructor be virtual?](https://devblogs.microsoft.com/oldnewthing/20040507-00/?p=39443).
* S. Meyers. *Effective C++ 3d ed.* pp 40-44. Addison-Wesley Professional, 2005.

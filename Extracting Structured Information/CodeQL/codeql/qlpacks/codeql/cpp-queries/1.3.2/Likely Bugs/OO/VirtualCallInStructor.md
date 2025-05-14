# Virtual call in constructor or destructor
This rule finds calls to virtual methods from within constructors and destructors. If the constructor of a derived class calls the constructor of its base type, the dynamic type of the object is the base type for the duration of the constructor call, so virtual method dispatch may yield unexpected results. The same holds for destructors.


## Recommendation
Carefully check the flagged calls to make sure they will behave as expected.


## Example

```cpp
class Base {
protected:
    Resource* resource;
public:
    virtual void init() {
        resource = createResource();
    }
    virtual void release() {
        freeResource(resource);
    }
};

class Derived: public Base {
    virtual void init() {
        resource = createResourceV2();
    }
    virtual void release() {
        freeResourceV2(resource);
    }
};

Base::Base() {
    this->init();
}
Base::~Base() {
    this->release();
}

int f() {
    // this will call Base::Base() and then Derived::Derived(), but this->init()
    // inBase::Base() will resolve to Base::init(), not Derived::init()
    // The reason for this is that when Base::Base is called, the object being
    // created is still of type Base (including the vtable)
    Derived* d = new Derived();
}

```

## References
* S. Meyer. *Effective C++ 3d ed.* pp 48-52. Addison Wesley. 2005. Excerpt available [here](http://www.artima.com/cppsource/nevercall.html).

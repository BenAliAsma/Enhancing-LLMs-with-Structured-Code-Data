# Non-virtual destructor in base class
This rule finds classes with virtual functions but no virtual destructor. Deleting a class without a virtual destructor will only call the destructor of the type of the pointer being deleted. This can cause a defect if the pointer type is a base type while the object instance is a derived type.


## Recommendation
Make sure that all classes with virtual functions also have a virtual destructor, especially if other classes derive from them.


## Example

```cpp
class Base {
public:
	Resource *p;
	Base() {
		p = createResource();
	}
	virtual void f() { //has virtual function
		//...
	}
	//...
	~Base() { //wrong: is non-virtual
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

	//will only call Base::~Base(), leaking the resource dp.
	//Change both destructors to virtual to ensure they are both called.
	delete b;
}

```

## References
* S. Meyers. *Effective C++ 3d ed.* pp 40-44. Addison-Wesley Professional, 2005.
* [When should your destructor be virtual?](https://devblogs.microsoft.com/oldnewthing/20040507-00/?p=39443)

# Inconsistent definition of copy constructor and assignment ('Rule of Two')
This rule finds classes that define a copy constructor or a copy assignment operator, but not both of them. The compiler generates default implementations for these functions, and since they deal with similar concerns it is likely that if the default implementation of one of them is not satisfactory, then neither is that of the other.

When a class defines a copy constructor or a copy assignment operator, but not both, this can cause unexpected behavior. The object initialization (that is, `Class c1 = c2`) may behave differently from object assignment (that is, `c1 = c2`).


## Recommendation
First, consider whether the user-defined member needs to be explicitly defined at all. If no user-defined copy constructor is provided for a class, the compiler will always attempt to generate a public copy constructor that recursively invokes the copy constructor of each field. If the existing user-defined copy constructor does exactly the same, it is most likely beneficial to delete it. The compiler-generated version may be more efficient, and it does not need to be manually maintained as fields are added and deleted.

If the user-defined member *does* need to exist, the other corresponding member should be defined too. It can be defined as defaulted (using `= default`) if the compiler-generated implementation is acceptable, or it can be defined as deleted (using `= delete`) if it should never be called.


## Example

```cpp
class C {
private:
	Other* other = NULL;
public:
	C(const C& copyFrom) {
		Other* newOther = new Other();
		*newOther = copyFrom.other;
		this->other = newOther;
	}

	//No operator=, by default will just copy the pointer other, will not create a new object
};

class D {
	Other* other = NULL;
public:
	D& operator=(D& rhs) {
		Other* newOther = new Other();
		*newOther = rhs.other;
		this->other = newOther;
		return *this;
	}

	//No copy constructor, will just copy the pointer other and not create a new object
};


```

## References
* [Rule of Three \[Wikipedia\]](http://en.wikipedia.org/wiki/Rule_of_three_(C%2B%2B_programming))
* [The Law of The Big Two](http://www.artima.com/cppsource/bigtwo.html)
* cppreference.com: [copy constructor](http://en.cppreference.com/w/cpp/language/copy_constructor) and [copy assignment](http://en.cppreference.com/w/cpp/language/copy_assignment)

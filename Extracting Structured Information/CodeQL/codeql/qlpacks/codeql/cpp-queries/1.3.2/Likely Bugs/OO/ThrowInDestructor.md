# Exception thrown in destructor
This rule finds exceptions thrown in destructors. This is dangerous: If the destructor is called during stack unwinding as part of the handling of another exception, throwing an additional exception will immediately terminate the program as per the C++ specification.


## Recommendation
Handle the error condition in a different way.


## Example

```cpp
class C {
public:
	//...
	~C(){
		if (error) {
			throw "Exception in destructor"; //wrong: exception thrown in destructor
		}
	}
};

void f() {
	C* c = new C();
	try {
		doOperation(c);
		delete c;
	} catch ( char * do_operation_exception) {
		delete c; //would immediately terminate program if C::~C throws an exception
	}
}

```

## References
* M. Cline, [C++ FAQ: How can I handle a destructor that fails?](https://isocpp.org/wiki/faq/exceptions#dtors-shouldnt-throw)
* B. Stroustrup. *The C++ Programming Language Special Edition* p 380. Addison Wesley. 2000.

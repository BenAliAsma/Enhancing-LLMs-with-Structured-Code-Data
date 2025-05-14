# Self assignment check
This rule finds copy assignment operators that deallocate memory but do not check for self assignment. This could lead to accessing an already freed memory location.

This rule is particularly important if the copy assignment operator puts the current object in an invalid state before getting a new value from the object on the right hand side.


## Recommendation
Copy assignment operator should check for self-assignment.


## Example
This example shows a copy assignment operator that fails to check for self assignment. The corrected version of the same operator is also included.


```cpp

class MyClass
{
public:
  MyClass() : data(new Data()) {}
  ~MyClass() {delete data;}

  MyClass &operator=(const MyClass &other)
  {
    delete data;
    data = other.data->clone(); // wrong: if other == *this, other.data has already been deleted!
    return *this;
  }

private:
  Data *data;
};

// If someone assigns a `MyClass` object to itself, the delete expression deletes both `this->data`
// and `other.data`, since `*this` and `other` are the same object. But the call to `clone` uses
// `*other.data`, which is no longer a valid object.  Fix this by adding a check:

class MyClass
{
public:
  MyClass() : data(new Data()) {}
  ~MyClass() {delete data;}

  MyClass &operator=(const MyClass &other)
  {
    if (this == &other) { return *this; } // correct
    delete data;
    data = other.data->clone();
    return *this;
  }

private:
  Data *data;
};

```

## References
* Standard C++ Foundation: [Assignment Operators](https://isocpp.org/wiki/faq/assignment-operators)
* Common Weakness Enumeration: [CWE-826](https://cwe.mitre.org/data/definitions/826.html).

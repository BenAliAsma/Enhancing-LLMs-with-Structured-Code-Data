# Type confusion
Certain casts in C and C++ place no restrictions on the target type. For example, C style casts such as `(MyClass*)p` allows the programmer to cast any pointer `p` to an expression of type `MyClass*`. If the runtime type of `p` turns out to be a type that's incompatible with `MyClass`, this results in undefined behavior.


## Recommendation
If possible, use `dynamic_cast` to safely cast between polymorphic types. If `dynamic_cast` is not an option, use `static_cast` to restrict the kinds of conversions that the compiler is allowed to perform. If C++ style casts is not an option, carefully check that all casts are safe.


## Example
Consider the following class hierachy where we define a base class `Shape` and two derived classes `Circle` and `Square` that are mutually incompatible:


```cpp
struct Shape {
  virtual ~Shape();

  virtual void draw() = 0;
};

struct Circle : public Shape {
  Circle();

  void draw() override {
    /* ... */
  }

  int getRadius();
};

struct Square : public Shape {
  Square();

  void draw() override {
    /* ... */
  }

  int getLength();
};

```
The following code demonstrates a type confusion vulnerability where the programmer assumes that the runtime type of `p` is always a `Square`. However, if `p` is a `Circle`, the cast will result in undefined behavior.


```cpp
void allocate_and_draw_bad() {
  Shape* shape = new Circle;
  // ...
  // BAD: Assumes that shape is always a Square
  Square* square = static_cast<Square*>(shape);
  int length = square->getLength();
}
```
The following code fixes the vulnerability by using `dynamic_cast` to safely cast between polymorphic types. If the cast fails, `dynamic_cast` returns a null pointer, which can be checked for and handled appropriately.


```cpp
void allocate_and_draw_good() {
  Shape* shape = new Circle;
  // ...
  // GOOD: Dynamically checks if shape is a Square
  Square* square = dynamic_cast<Square*>(shape);
  if(square) {
    int length = square->getLength();
  } else {
    // handle error
  }
}
```

## References
* Microsoft Learn: [Type conversions and type safety](https://learn.microsoft.com/en-us/cpp/cpp/type-conversions-and-type-safety-modern-cpp).
* Common Weakness Enumeration: [CWE-843](https://cwe.mitre.org/data/definitions/843.html).

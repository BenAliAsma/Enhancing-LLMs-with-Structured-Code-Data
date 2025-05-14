# Redefined default parameter
Inherited default parameters should not be redefined because this obscures the behavior of the code. Default values are statically bound and when they are redefined in dynamically bound function calls this reduces readability of the code, increasing the risk of introducing defects.

For example, while C++ dynamically binds virtual methods, the default parameters of those methods are statically bound. Hence, the draw() method of the derived type (Circle), if referenced through a base type pointer (Shape \*), will be invoked with the default parameters of the base type (Shape).


```cpp
enum Shape_color { red, green, blue };
class Shape 
{
  public:
    virtual void draw (Shape_color color = green) const;
    ...
}
class Circle : public Shape 
{
  public:
    virtual void draw (Shape_color  color = red) const;
    ...
}
void fun()
{
  Shape* sp;

  sp = new Circle;
  sp->draw ();      // Invokes Circle::draw(green) even though the default
}				   	// parameter for Circle is red.

```

## References
* AV Rule 95, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* Scott Meyers. Item 38, Effective C++: 50 Specific Ways to Improve Your Programs and Design, 2nd Edition. Addison-Wesley, 1998.

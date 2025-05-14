# Missing call to `__init__` during object initialization
Python, unlike statically typed languages such as Java, allows complete freedom when calling methods during object initialization. However, standard object-oriented principles apply to Python classes using deep inheritance hierarchies. Therefore the developer has responsibility for ensuring that objects are properly initialized when there are multiple `__init__` methods that need to be called.

If the `__init__` method of a superclass is not called during object initialization it is likely that that object will end up in an incorrect state.

A call to the `__init__` method of a superclass during object initialization may be omitted:

* When a subclass calls the `__init__` method of the wrong class.
* When a call to the `__init__` method of one its base classes is omitted.
* When multiple inheritance is used and a class inherits from several base classes, and at least one of those does not use `super()` in its own `__init__` method.

## Recommendation
Either be careful to explicitly call the `__init__` of the correct base class, or use `super()` throughout the inheritance hierarchy.

Alternatively refactor one or more of the classes to use composition rather than inheritance.


## Example
In this example, explicit calls to `__init__` are used, but `SportsCar` erroneously calls `Vehicle.__init__`. This is fixed in `FixedSportsCar` by calling `Car.__init__`.


```python

class Vehicle(object):
    
    def __init__(self):
        self.mobile = True
        
class Car(Vehicle):
    
    def __init__(self):
        Vehicle.__init__(self)
        self.car_init()
        
#Car.__init__ is missed out.
class SportsCar(Car, Vehicle):
    
    def __init__(self):
        Vehicle.__init__(self)
        self.sports_car_init()
        
#Fix SportsCar by calling Car.__init__
class FixedSportsCar(Car, Vehicle):
    
    def __init__(self):
        Car.__init__(self)
        self.sports_car_init()
        

```

## References
* Python Tutorial: [Classes](https://docs.python.org/2/tutorial/classes.html).
* Python Standard Library: [super](https://docs.python.org/2/library/functions.html#super).
* Artima Developer: [Things to Know About Python Super](http://www.artima.com/weblogs/viewpost.jsp?thread=236275).
* Wikipedia: [Composition over inheritance](http://en.wikipedia.org/wiki/Composition_over_inheritance).

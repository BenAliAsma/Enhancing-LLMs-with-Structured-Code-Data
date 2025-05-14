# Missing call to `__del__` during object destruction
Python, unlike statically typed languages such as Java, allows complete freedom when calling methods during object destruction. However, standard object-oriented principles apply to Python classes using deep inheritance hierarchies. Therefore the developer has responsibility for ensuring that objects are properly cleaned up when there are multiple `__del__` methods that need to be called.

If the `__del__` method of a superclass is not called during object destruction it is likely that that resources may be leaked.

A call to the `__del__` method of a superclass during object destruction may be omitted:

* When a subclass calls the `__del__` method of the wrong class.
* When a call to the `__del__` method of one its base classes is omitted.

## Recommendation
Either be careful to explicitly call the `__del__` of the correct base class, or use `super()` throughout the inheritance hierarchy.

Alternatively refactor one or more of the classes to use composition rather than inheritance.


## Example
In this example, explicit calls to `__del__` are used, but `SportsCar` erroneously calls `Vehicle.__del__`. This is fixed in `FixedSportsCar` by calling `Car.__del__`.


```python

class Vehicle(object):
    
    def __del__(self):
        recycle(self.base_parts)
        
class Car(Vehicle):
    
    def __del__(self):
        recycle(self.car_parts)
        Vehicle.__del__(self)
        
#Car.__del__ is missed out.
class SportsCar(Car, Vehicle):
    
    def __del__(self):
        recycle(self.sports_car_parts)
        Vehicle.__del__(self)
        
#Fix SportsCar by calling Car.__del__
class FixedSportsCar(Car, Vehicle):
    
    def __del__(self):
        recycle(self.sports_car_parts)
        Car.__del__(self)
        

```

## References
* Python Tutorial: [Classes](https://docs.python.org/2/tutorial/classes.html).
* Python Standard Library: [super](https://docs.python.org/2/library/functions.html#super).
* Artima Developer: [Things to Know About Python Super](http://www.artima.com/weblogs/viewpost.jsp?thread=236275).
* Wikipedia: [Composition over inheritance](http://en.wikipedia.org/wiki/Composition_over_inheritance).

# Multiple calls to `__init__` during object initialization
Python, unlike statically typed languages such as Java, allows complete freedom when calling methods during object initialization. However, standard object-oriented principles apply to Python classes using deep inheritance hierarchies. Therefore the developer has responsibility for ensuring that objects are properly initialized when there are multiple `__init__` methods that need to be called.

Calling an `__init__` method more than once during object initialization risks the object being incorrectly initialized. It is unlikely that the relevant `__init__` method is designed to be called more than once.

There are a number of ways that an `__init__` method may be be called more than once.

* There may be more than one explicit call to the method in the hierarchy of `__init__` methods.
* A class using multiple inheritance directly calls the `__init__` methods of its base types. One or more of those base types uses `super()` to pass down the inheritance chain.

## Recommendation
Either be careful not to explicitly call an `__init__` method more than once, or use `super()` throughout the inheritance hierarchy.

Alternatively refactor one or more of the classes to use composition rather than inheritance.


## Example
In the first example, explicit calls to `__init__` are used, but `SportsCar` erroneously calls both `Vehicle.__init__` and `Car.__init__`. This can be fixed by removing the call to `Vehicle.__init__`, as shown in `FixedSportsCar`.


```python
#Calling a method multiple times by using explicit calls when a base inherits from other base
class Vehicle(object):
    
    def __init__(self):
        self.mobile = True
        
class Car(Vehicle):
    
    def __init__(self):
        Vehicle.__init__(self)
        self.car_init()
        
    def car_init(self):
        pass
    
class SportsCar(Car, Vehicle):
    
    # Vehicle.__init__ will get called twice
    def __init__(self):
        Vehicle.__init__(self)
        Car.__init__(self)
        self.sports_car_init()
        
    def sports_car_init(self):
        pass
        
#Fix SportsCar by only calling Car.__init__
class FixedSportsCar(Car, Vehicle):
    
    def __init__(self):
        Car.__init__(self)
        self.sports_car_init()
        
    def sports_car_init(self):
        pass
 

```
In the second example, there is a mixture of explicit calls to `__init__` and calls using `super()`. To fix this example, `super()` should be used throughout.


```python

#Calling a method multiple times by using explicit calls when a base uses super()
class Vehicle(object):
     
    def __init__(self):
        super(Vehicle, self).__init__()
        self.mobile = True
        
class Car(Vehicle):
    
    def __init__(self):
        super(Car, self).__init__()
        self.car_init()
        
    def car_init(self):
        pass
        
class SportsCar(Car, Vehicle):
    
    # Vehicle.__init__ will get called twice
    def __init__(self):
        Vehicle.__init__(self)
        Car.__init__(self)
        self.sports_car_init()
        
    def sports_car_init(self):
        pass
        
#Fix SportsCar by using super()
class FixedSportsCar(Car, Vehicle):
    
    def __init__(self):
        super(SportsCar, self).__init__()
        self.sports_car_init()
        
    def sports_car_init(self):
        pass


```

## References
* Python Tutorial: [Classes](https://docs.python.org/2/tutorial/classes.html).
* Python Standard Library: [super](https://docs.python.org/2/library/functions.html#super).
* Artima Developer: [Things to Know About Python Super](http://www.artima.com/weblogs/viewpost.jsp?thread=236275).
* Wikipedia: [Composition over inheritance](http://en.wikipedia.org/wiki/Composition_over_inheritance).

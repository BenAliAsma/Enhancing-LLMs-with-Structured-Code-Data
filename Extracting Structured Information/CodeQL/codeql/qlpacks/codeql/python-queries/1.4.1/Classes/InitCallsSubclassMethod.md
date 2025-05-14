# `__init__` method calls overridden method
When an instance of a class is initialized, the super-class state should be fully initialized before it becomes visible to the subclass. Calling methods of the subclass in the superclass' `__init__` method violates this important invariant.


## Recommendation
Do not use methods that are subclassed in the construction of an object. For simpler cases move the initialization into the superclass' `__init__` method, preventing it being overridden. Additional initialization of subclass should be done in the `__init__` method of the subclass. For more complex cases, it is advisable to use a static method or function to manage object creation.

Alternatively, avoid inheritance altogether using composition instead.


## Example

```python
#Superclass __init__ calls subclass method

class Super(object):

    def __init__(self, arg):
        self._state = "Not OK"
        self.set_up(arg)
        self._state = "OK"

    def set_up(self, arg):
        "Do some set up"

class Sub(Super):

    def __init__(self, arg):
        Super.__init__(self, arg)
        self.important_state = "OK"

    def set_up(self, arg):
        Super.set_up(self, arg)
        "Do some more set up" # Dangerous as self._state is "Not OK"


#Improved version with inheritance:

class Super(object):

    def __init__(self, arg):
        self._state = "Not OK"
        self.super_set_up(arg)
        self._state = "OK"

    def super_set_up(self, arg):
        "Do some set up"


class Sub(Super):

    def __init__(self, arg):
        Super.__init__(self, arg)
        self.sub_set_up(self, arg)
        self.important_state = "OK"


    def sub_set_up(self, arg):
        "Do some more set up"



```

## References
* CERT Secure Coding: [ Rule MET05-J](https://www.securecoding.cert.org/confluence/display/java/MET05-J.+Ensure+that+constructors+do+not+call+overridable+methods). Although this is a Java rule it applies to most object-oriented languages.
* Python Standard Library: [Static methods](http://docs.python.org/library/functions.html#staticmethod).
* Wikipedia: [Composition over inheritance](http://en.wikipedia.org/wiki/Composition_over_inheritance).

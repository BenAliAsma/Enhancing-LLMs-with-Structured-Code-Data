# Inconsistent equality and hashing
In order to conform to the object model, classes that define their own equality method should also define their own hash method, or be unhashable. If the hash method is not defined then the `hash` of the super class is used. This is unlikely to result in the expected behavior.

A class can be made unhashable by setting its `__hash__` attribute to `None`.

In Python 3, if you define a class-level equality method and omit a `__hash__` method then the class is automatically marked as unhashable.


## Recommendation
When you define an `__eq__` method for a class, remember to implement a `__hash__` method or set `__hash__ = None`.


## Example
In the following example the `Point` class defines an equality method but no hash method. If hash is called on this class then the hash method defined for `object` is used. This is unlikely to give the required behavior. The `PointUpdated` class is better as it defines both an equality and a hash method. If `Point` was not to be used in `dict`s or `set`s, then it could be defined as `UnhashablePoint` below.

To comply fully with the object model this class should also define an inequality method (identified by a separate rule).


```python
# Incorrect: equality method defined but class contains no hash method
class Point(object):

    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __repr__(self):
        return 'Point(%r, %r)' % (self._x, self._y)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self._x == other._x and self._y == other._y


# Improved: equality and hash method defined (inequality method still missing)
class PointUpdated(object):

    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __repr__(self):
        return 'Point(%r, %r)' % (self._x, self._y)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self._x == other._x and self._y == other._y

    def __hash__(self):  
        return hash(self._x) ^ hash(self._y)

# Improved: equality method defined and class instances made unhashable
class UnhashablePoint(object):

    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __repr__(self):
        return 'Point(%r, %r)' % (self._x, self._y)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self._x == other._x and self._y == other._y

    #Tell the interpreter that instances of this class cannot be hashed
    __hash__ = None


```

## References
* Python Language Reference: [object.__hash__](http://docs.python.org/reference/datamodel.html#object.__hash__).
* Python Glossary: [hashable](http://docs.python.org/2/glossary.html#term-hashable).
* Common Weakness Enumeration: [CWE-581](https://cwe.mitre.org/data/definitions/581.html).

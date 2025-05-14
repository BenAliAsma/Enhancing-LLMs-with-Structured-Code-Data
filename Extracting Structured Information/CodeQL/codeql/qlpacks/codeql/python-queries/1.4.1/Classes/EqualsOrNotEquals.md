# Inconsistent equality and inequality
In order to conform to the object model, classes should define either no equality methods, or both an equality and an inequality method. If only one of `__eq__` or `__ne__` is defined then the method from the super class is used. This is unlikely to result in the expected behavior.


## Recommendation
When you define an equality or an inequality method for a class, remember to implement both an `__eq__` method and an `__ne__` method.


## Example
In the following example the `PointOriginal` class defines an equality method but no inequality method. If this class is tested for inequality then a type error will be raised. The `PointUpdated` class is better as it defines both an equality and an inequality method. To comply fully with the object model this class should also define a hash method (identified by a separate rule).


```python
class PointOriginal(object):

    def __init__(self, x, y):
        self._x, x
        self._y = y

    def __repr__(self):
        return 'Point(%r, %r)' % (self._x, self._y)

    def __eq__(self, other):  # Incorrect: equality is defined but inequality is not
        if not isinstance(other, Point):
            return False
        return self._x == other._x and self._y == other._y


class PointUpdated(object):

    def __init__(self, x, y):
        self._x, x
        self._y = y

    def __repr__(self):
        return 'Point(%r, %r)' % (self._x, self._y)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self._x == other._x and self._y == other._y

    def __ne__(self, other):  # Improved: equality and inequality method defined (hash method still missing)
        return not self == other


```

## References
* Python Language Reference: [object.__ne__](http://docs.python.org/2/reference/datamodel.html#object.__ne__), [Comparisons](http://docs.python.org/2/reference/expressions.html#comparisons).

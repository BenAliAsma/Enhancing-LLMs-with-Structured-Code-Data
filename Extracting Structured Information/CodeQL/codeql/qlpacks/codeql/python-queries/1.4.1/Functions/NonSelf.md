# First parameter of a method is not named 'self'
Normal methods should have at least one parameter and the first parameter should be called `self`. This makes the purpose of the parameter clear to other developers.


## Recommendation
If there is at least one parameter, then change the name of the first parameter to `self` as recommended by the style guidelines in PEP 8.

If there are no parameters, then it cannot be a normal method. It may need to be marked as a `staticmethod` or it could be moved out of the class as a normal function.


## Example
The following methods can both be used to assign values to variables in a `point` object. The second method makes the association clearer because the `self` parameter is used.


```python
class Point:
    def __init__(val, x, y):  # first parameter is mis-named 'val'
        val._x = x
        val._y = y

class Point2:
    def __init__(self, x, y):  # first parameter is correctly named 'self'
        self._x = x
        self._y = y
```

## References
* Python PEP 8: [Function and method arguments](http://www.python.org/dev/peps/pep-0008/#function-and-method-arguments).
* Python Tutorial: [Classes](http://docs.python.org/2/tutorial/classes.html).

# First parameter of a class method is not named 'cls'
The first parameter of a class method, a new method or any metaclass method should be called `cls`. This makes the purpose of the parameter clear to other developers.


## Recommendation
Change the name of the first parameter to `cls` as recommended by the style guidelines in PEP 8.


## Example
In the example, the first parameter to `make()` is `klass` which should be changed to `cls` for ease of comprehension.


```python
class Entry(object):
    @classmethod
    def make(klass):
        return Entry()

```

## References
* Python PEP 8: [Function and method arguments](http://www.python.org/dev/peps/pep-0008/#function-and-method-arguments).
* Python Tutorial: [Classes](http://docs.python.org/2/tutorial/classes.html).

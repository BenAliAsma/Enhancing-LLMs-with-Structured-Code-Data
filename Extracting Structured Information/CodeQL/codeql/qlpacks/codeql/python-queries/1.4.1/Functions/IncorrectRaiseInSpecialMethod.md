# Non-standard exception raised in special method
User-defined classes interact with the Python virtual machine via special methods (also called "magic methods"). For example, for a class to support addition it must implement the `__add__` and `__radd__` special methods. When the expression `a + b` is evaluated the Python virtual machine will call `type(a).__add__(a, b)` and if that is not implemented it will call `type(b).__radd__(b, a)`.

Since the virtual machine calls these special methods for common expressions, users of the class will expect these operations to raise standard exceptions. For example, users would expect that the expression `a.b` might raise an `AttributeError` if the object `a` does not have an attribute `b`. If a `KeyError` were raised instead, then this would be unexpected and may break code that expected an `AttributeError`, but not a `KeyError`.

Therefore, if a method is unable to perform the expected operation then its response should conform to the standard protocol, described below.

* Attribute access, `a.b`: Raise `AttributeError`
* Arithmetic operations, `a + b`: Do not raise an exception, return `NotImplemented` instead.
* Indexing, `a[b]`: Raise `KeyError`.
* Hashing, `hash(a)`: Use `__hash__ = None` to indicate that an object is unhashable.
* Equality methods, `a != b`: Never raise an exception, always return `True` or `False`.
* Ordering comparison methods, `a < b`: Raise a `TypeError` if the objects cannot be ordered.
* Most others: Ideally, do not implement the method at all, otherwise raise `TypeError` to indicate that the operation is unsupported.

## Recommendation
If the method is meant to be abstract, then declare it so using the `@abstractmethod` decorator. Otherwise, either remove the method or ensure that the method raises an exception of the correct type.


## Example
This example shows two unhashable classes. The first class is unhashable in a non-standard way which may cause maintenance problems. The second, corrected, class uses the standard idiom for unhashable classes.


```python
#Incorrect unhashable class
class MyMutableThing(object):
    
    def __init__(self):
        pass
    
    def __hash__(self):
        raise NotImplementedError("%r is unhashable" % self)

#Make class unhashable in the standard way
class MyCorrectMutableThing(object):
    
    def __init__(self):
        pass
    
    __hash__ = None

```
In this example, the first class is implicitly abstract; the `__add__` method is unimplemented, presumably with the expectation that it will be implemented by sub-classes. The second class makes this explicit with an `@abstractmethod` decoration on the unimplemented `__add__` method.


```python
    
#Abstract base class, but don't declare it.
class ImplicitAbstractClass(object):
    
    def __add__(self, other):
        raise NotImplementedError()
    
#Make abstractness explicit.
class ExplicitAbstractClass:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError()
 

```
In this last example, the first class implements a collection backed by the file store. However, should an `IOError` be raised in the `__getitem__` it will propagate to the caller. The second class handles any `IOError` by reraising a `KeyError` which is the standard exception for the `__getitem__` method.


```python
    
#Incorrect file-backed table
class FileBackedTable(object):
    
    def __getitem__(self, key):
        if key not in self.index:
            raise IOError("Key '%s' not in table" % key)
        else:
            #May raise an IOError
            return self.backing.get_row(key)
        
#Correct by transforming exception
class ObjectLikeFileBackedTable(object):
    
    def get_from_key(self, key):
        if key not in self.index:
            raise IOError("Key '%s' not in table" % key)
        else:
            #May raise an IOError
            return self.backing.get_row(key)
    
    def __getitem__(self, key):
        try:
            return self.get_from_key(key)
        except IOError:
            raise KeyError(key)
                           

```

## References
* Python Language Reference: [Special Method Names](http://docs.python.org/dev/reference/datamodel.html#special-method-names).
* Python Library Reference: [Exceptions](https://docs.python.org/2/library/exceptions.html).

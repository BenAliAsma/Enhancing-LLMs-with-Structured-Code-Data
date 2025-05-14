# Iterator does not return self from `__iter__` method
The `__iter__` method of an iterator should return self. This is important so that iterators can be used as sequences in any context that expect a sequence. To do so requires that `__iter__` is idempotent on iterators.

Note that sequences and mapping should return a new iterator, it is just the returned iterator that must obey this constraint.


## Recommendation
Make the `__iter__` return self unless the class should not be an iterator, in which case rename the `next` (Python 2) or `__next__` (Python 3) to something else.


## Example
In this example the `Counter` class's `__iter__` method does not return self (or even an iterator). This will cause the program to fail when anyone attempts to use the iterator in a `for` loop or `in` statement.


```python
class MyRange(object):
    def __init__(self, low, high):
        self.current = low
        self.high = high

    def __iter__(self):
        return self.current

    def next(self):
        if self.current > self.high:
            raise StopIteration
        self.current += 1
        return self.current - 1
```

## References
* Python Language Reference: [object.__iter__](http://docs.python.org/2.7/reference/datamodel.html#object.__iter__).
* Python Standard Library: [Iterators](http://docs.python.org/2/library/stdtypes.html#typeiter).

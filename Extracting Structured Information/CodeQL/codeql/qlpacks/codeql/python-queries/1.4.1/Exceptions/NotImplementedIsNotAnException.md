# NotImplemented is not an Exception
`NotImplemented` is not an Exception, but is often mistakenly used in place of `NotImplementedError`. Executing `raise NotImplemented` or `raise NotImplemented()` will raise a `TypeError`. When `raise NotImplemented` is used to mark code that is genuinely never called, this mistake is benign. However, should it be called, then a `TypeError` will be raised rather than the expected `NotImplemented`, which might make debugging the issue difficult.

The correct use of `NotImplemented` is to implement binary operators. Code that is not intended to be called should raise `NotImplementedError`.


## Recommendation
Replace uses of `NotImplemented` with `NotImplementedError`.


## Example
In the example below, the method `wrong` will incorrectly raise a `TypeError` when called. The method `right` will raise a `NotImplementedError`.


```python

class Abstract(object):

    def wrong(self):
        # Will raise a TypeError
        raise NotImplemented()

    def right(self):
        raise NotImplementedError()

```

## References
* Python Language Reference: [The NotImplementedError exception](https://docs.python.org/library/exceptions.html#NotImplementedError).
* Python Language Reference: [Emulating numeric types](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types).

# Signature mismatch in overriding method
There are one (or more) legal parameters for an overridden method that are not legal for an overriding method. This will cause an error when the overriding method is called with a number of parameters that is legal for the overridden method. This violates the Liskov substitution principle.


## Recommendation
Ensure that the overriding method accepts all the parameters that are legal for overridden method.


## Example
In this example there is a mismatch between the legal parameters for the base class method `(self, source, filename, symbol)` and the extension method `(self, source)`. The extension method can be used to override the base method as long as values are not specified for the `filename` and `symbol` parameters. If the extension method was passed the additional parameters accepted by the base method then an error would occur.


```python

# Base class method
def runsource(self, source, filename="<input>", symbol="single"):
    ... # Definition
    
    
# Extend base class method
def runsource(self, source):
    ... # Definition
```
The extension method should be updated to support the `filename` and `symbol` parameters supported by the overridden method.


## References
* Wikipedia: [Liskov Substitution Principle](http://en.wikipedia.org/wiki/Liskov_substitution_principle), [Method overriding](http://en.wikipedia.org/wiki/Method_overriding#Python).

# Number of parameters per method/constructor/operator
This metric measures the number of parameters for each method. If a method has a large number of parameters it becomes difficult to call. It is also an indication that the method may be trying to do too much.


## Recommendation
If the parameters of the method are logically related (for example different lines of an address) it may be best to create a new class to contain them, create an instance of that class and pass the class to the method.

If the method only uses a parameter in a short part of its code then possibly this section should be split off into another method.

Another possibility is that the parameters are not actually used at all. They may have been added when the method was first written and then it was discovered they were not needed but they were not removed from the method signature. In these cases it should be reasonably easy to remove the parameter and refactor calls to the method appropriately.

If the method is part of a public interface it may be difficult to modify the signature so it is important to design interfaces correctly the first time.


## References
* Martin Fowler. *Refactoring*. Addison-Wesley. 1999.

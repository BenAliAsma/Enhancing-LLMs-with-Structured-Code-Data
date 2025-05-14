# Response per class
This metric measures the number of different functions that can be executed by an object of that class in response to a message.

Classes with a high response metric can be difficult to understand and test, because you have to read through all the functions that can possibly be called in order to fully understand what's going on.


## Recommendation
Generally speaking, when a class has a high response metric, it is because it contains functions that individually make large numbers of calls and/or because it has high efferent coupling. The solution is therefore to fix these underlying problems, and the class's response will decrease accordingly.


## References
* S. R. Chidamber and C. F. Kemerer. *A metrics suite for object-oriented design*. IEEE Transactions on Software Engineering, 20(6):476-493, 1994.

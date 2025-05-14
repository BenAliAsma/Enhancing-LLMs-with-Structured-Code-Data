# Outgoing file dependencies
This metric measures the number of outgoing dependencies for each compilation unit, i.e. the number of other units on which this compilation unit depends.

Compilation units that depend on many other units are typically subject to more frequent recompilation, because they have to be recompiled every time one of the many units on which they depend changes. It is particularly problematic if such a compilation unit is also heavily depended on by other units, because any minor change to one of its dependencies will trigger a ripple effect of recompilation throughout the code: such units are known as hubs, and can be responsible for greatly increasing the time needed for incremental recompilation.


## Recommendation
Unit-level efferent coupling can be reduced by splitting the unit into pieces along its dependency fault lines, as illustrated by the following diagrams. Initially, the compilation unit has many incoming and outgoing dependencies, but it is not the case that the entire unit depends on every dependency. For example, sub-unit `Y` only depends on `O2` and `O4` (note that `Y` is not necessarily a class, just a part of the compilation unit):

<table> <tbody><tr> <td><img></img></td> </tr> <tr> <td>Before</td> </tr> </tbody></table>
This suggests the following refactoring, where the unit containing `X`, `Y` and `Z` is split into three new units, each with fewer dependencies. Notice how the efferent coupling of each of the new units is significantly lower than that of the original unit:

<table> <tbody><tr> <td><img></img></td> </tr> <tr> <td>After</td> </tr> </tbody></table>
Performing this refactoring clearly relies on an ability to determine suitable places at which to partition the compilation unit. This can often be done by simple inspection of the code, but for more complicated units there are scientific approaches as well. As a simple example of this, consider a compilation unit that contains multiple classes. This could be partitioned by visualizing the dependencies of the various classes and grouping those with the same dependencies. Each group would then be turned into a new compilation unit. (Of course, for classes it is often best to just have a compilation unit for each class; this example is merely intended to illustrate how one might go about partitioning a unit in a mechanical way.)


## References
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.

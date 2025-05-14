# Lack of type cohesion (HS)
This metric provides an indication of the lack of cohesion of a reference type, using a method proposed by Brian Henderson-Sellers in 1996. The idea behind measuring a type's cohesion is that most methods in well-designed classes will access the same fields. Types that exhibit a lack of cohesion are often trying to take on multiple responsibilities, and should be split into several smaller classes.

Various measures of lack of cohesion have been proposed: while the basic intuition is simple, the precise way to measure this property has been the subject of intense debate. Rather than getting involved in this debate, more than one such lack of cohesion measure is provided for comparison purposes.

The Henderson-Sellers version of lack of cohesion can be defined as follows. Let `M` be the set of methods in a class, `F` be the set of fields in the class, `r(f)` be the number of methods that access field `f`, and `ar` be the mean of `r(f)` over `f` in `F`, then the lack of cohesion measure `LCOM` can be defined as:

```

LCOM = (ar - |M|) / (1 - |M|)

```
As per \[Walton\], we restrict `M` to methods that read some field in the class, and `F` to fields that are read by some method in the class. High values of `LCOM` indicate a worrisome lack of cohesion. The precise value of the metric for which warnings are issued is configurable, but as a rough indication, an `LCOM` of `0.95` or more may give you cause for concern.


## Recommendation
Types generally lack cohesion because they are taking on more responsibilities than they should (see \[Martin\] for more on responsibilities). In general, the solution is to identify each of the different responsibilities the class is taking on, and split them out into multiple classes, e.g. using the 'Extract Class' refactoring from \[Fowler\].


## References
* M. Fowler. *Refactoring* pp. 65, 122-5. Addison-Wesley, 1999.
* B. Henderson-Sellers. *Object-Oriented Metrics: Measures of Complexity*. Prentice-Hall, 1996.
* R. Martin. *Agile Software Development: Principles, Patterns, and Practices* Chapter 8 - SRP: The Single-Responsibility Principle. Pearson Education, 2003.
* O. de Moor et al. *Keynote Address: .QL for Source Code Analysis*. Proceedings of the 7th IEEE International Working Conference on Source Code Analysis and Manipulation, 2007.

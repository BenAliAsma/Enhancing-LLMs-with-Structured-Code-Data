# Lack of cohesion per class (LCOM-CK)
This metric provides an indication of the lack of cohesion of a class, using a method proposed by Chidamber and Kemerer in 1994. The idea behind measuring a class's cohesion is that most functions in well-designed classes will access the same fields. Types that exhibit a lack of cohesion are often trying to take on multiple responsibilities, and should be split into several smaller classes.

Various measures of lack of cohesion have been proposed: while the basic intuition is simple, the precise way to measure this property has been the subject of intense debate. Rather than getting involved in this debate, more than one such lack of cohesion measure is provided for comparison purposes.

The Chidamber and Kemerer version of lack of cohesion inspects pairs of functions. If there are many pairs that access the same data, then the class is cohesive. On the other hand, if there are many pairs that do not access any common data, then the class is not cohesive. More precisely, if we let `n1` be the number of pairs of distinct functions in a class that do not have at least one commonly-accessed field, and let `n2` be the number of pairs of distinct functions in a class that do have at least one commonly-accessed field, then the lack of cohesion measure `LCOM` can be defined as:

```

LCOM = max((n1 - n2) / 2, 0)

```
High values of `LCOM` indicate a worrisome lack of cohesion. The precise value of the metric for which warnings are issued is configurable, but as a rough indication, an `LCOM` of `500` or more may give you cause for concern.


## Recommendation
Classes generally lack cohesion because they are taking on more responsibilities than they should (see \[Martin\] for more on responsibilities). In general, the solution is to identify each of the different responsibilities the class is taking on, and split them out into multiple classes, e.g. using the 'Extract Class' refactoring from \[Fowler\].


## References
* S. R. Chidamber and C. F. Kemerer. *A metrics suite for object-oriented design*. IEEE Transactions on Software Engineering, 20(6):476-493, 1994.
* M. Fowler. *Refactoring* pp. 65, 122-5. Addison-Wesley, 1999.
* R. Martin. [The Single Responsibility Principle](https://drive.google.com/file/d/0ByOwmqah_nuGNHEtcU5OekdDMkk/view). Published online.
* O. de Moor et al. *Keynote Address: .QL for Source Code Analysis*. Proceedings of the 7th IEEE International Working Conference on Source Code Analysis and Manipulation, 2007.

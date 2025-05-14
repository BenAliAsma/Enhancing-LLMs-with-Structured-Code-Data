# Block with too many statements
This rule finds blocks of code that have too many complex statements, such as branching statements (`if`, `switch`), and loops (`for`, `while`).

Blocks with too many consecutive statements are candidates for refactoring. Only complex statements are counted here (eg. for, while, switch ...). The top-level logic will be clearer if each complex statement is extracted to a function.


## Recommendation
It is often the case that each consecutive complex statement performs a dedicated separate task. It is a very common case that each complex statement is actually commented with a description of the task. Extract each such task into its own function for improved readability and to promote reuse.


## References
* M. Fowler. *Refactoring* Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* Microsoft Patterns &amp; Practices Team. [Architectural Patterns and Styles](http://msdn.microsoft.com/en-us/library/ee658117.aspx) *Microsoft Application Architecture Guide, 2nd Edition.* Microsoft Press, 2009.

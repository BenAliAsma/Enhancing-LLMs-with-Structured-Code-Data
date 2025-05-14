# Lines of comment per method
This metric measures the (absolute) number of comment lines per method. The value of this metric should be considered in the context of the complexity of the method: a simple method with little documentation may be fine but a more complex method may be difficult to understand and maintain.


## Recommendation
Consider if the method needs more documentation. Simple methods might not but most methods should have at least a comment explaining their purpose. If the method is particularly long and complicated it might be a good idea to extract parts of it into different methods. If these methods are given meaningful names then readability will be increased without the need for more comments.


## References
* Jeff Atwood. [Avoiding Undocumentation](http://www.codinghorror.com/blog/2005/11/avoiding-undocumentation.html). 2005.
* Steve McConnell. *Code Complete*. 2nd Edition. Microsoft Press. 2004.

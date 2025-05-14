# Number of classes
This metric measures the number of classes in each file. Files with a large number of classes in are more difficult to read and manage. The structure of the project is not reflected in the organization of files which can make classes more difficult to find. They can also cause problems with version control systems by increasing the likelihood that two developers work on the same file at once. That said, there are sometimes advantages to grouping classes together into the same file so care should be taken when evaluating this metric.


## Recommendation
Consider whether the classes are logically related. If they are not then it makes sense to put them in separate files. If the main class in the file contains some large nested classes then consider moving them to their own files using partial classes.


## References
* MSDN. C\# Programming Guide. [Partial Class Definitions](http://msdn.microsoft.com/en-us/library/wa80x488%28VS.80%29.aspx).

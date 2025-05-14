# Mostly duplicate class
When most of the methods in one class are duplicated in one or more other classes, the classes themselves are regarded as *mostly duplicate*.

Code duplication in general is highly undesirable for a range of reasons. The artificially inflated amount of code is more difficult to understand, and sequences of similar but subtly different lines can mask the real purpose or intention behind them. Also, there is always a risk that only one of several copies of the code is updated to address a defect or add a feature.


## Recommendation
Although completely duplicated classes are rare, they are usually a sign of a simple oversight (or deliberate copy/paste) by a developer. Usually the required solution is to remove all but one of them.

It is more common to see duplication of many methods between two classes, leaving just a few that are actually different. Decide whether the differences are intended or the result of an inconsistent update to one of the copies:

* If the two classes serve different purposes but many of their methods are duplicated, this indicates that there is a missing level of abstraction. Introducing a common super-class to define the common methods is likely to prevent many problems in the long term. Modern IDEs may provide refactoring support for this sort of issue, usually with the names "Pull up" or "Extract supertype".
* If the two classes serve the same purpose and are different only as a result of inconsistent updates then treat the classes as completely duplicate. Determine the most up-to-date and correct version of the code and eliminate all near duplicates.

## References
* E. Juergens, F. Deissenboeck, B. Hummel, S. Wagner. *Do code clones matter?* Proceedings of the 31st International Conference on Software Engineering, 485-495, 2009.

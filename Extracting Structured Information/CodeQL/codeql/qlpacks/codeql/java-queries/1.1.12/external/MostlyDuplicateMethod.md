# Mostly duplicate method
When most of the lines in one method are duplicated in one or more other methods, the methods themselves are regarded as *mostly duplicate* or *similar*.

Code duplication in general is highly undesirable for a range of reasons. The artificially inflated amount of code is more difficult to understand, and sequences of similar but subtly different lines can mask the real purpose or intention behind them. Also, there is always a risk that only one of several copies of the code is updated to address a defect or add a feature.


## Recommendation
Although completely duplicated methods are rare, they are usually a sign of a simple oversight (or deliberate copy/paste) by a developer. Usually the required solution is to remove all but one of them.

It is more common to see duplication of many lines between two methods, leaving just a few that are actually different. Decide whether the differences are intended or the result of an inconsistent update to one of the copies.

* If the two methods serve different purposes but many of their lines are duplicated, this indicates that there is a missing level of abstraction. Look for ways of encapsulating the commonality and sharing it while retaining the differences in functionality. Perhaps the method can be moved to a single place and given an additional parameter, allowing it to cover all use cases. Alternatively, there may be a common pre-processing or post-processing step that can be extracted to its own (shared) method, leaving only the specific parts in the existing methods. Modern IDEs may provide refactoring support for this sort of issue, usually with the names "Extract method", "Change method signature", "Pull up" or "Extract supertype".
* If the two methods serve the same purpose and are different only as a result of inconsistent updates then treat the methods as completely duplicate. Determine the most up-to-date and correct version of the code and eliminate all near duplicates. Callers of the removed methods should be updated to call the remaining method instead.

## References
* E. Juergens, F. Deissenboeck, B. Hummel, S. Wagner. *Do code clones matter?* Proceedings of the 31st International Conference on Software Engineering, 485-495, 2009.

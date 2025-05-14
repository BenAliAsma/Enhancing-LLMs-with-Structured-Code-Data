# Duplicate method
A method should never be duplicated exactly in several places in the code. The severity of this problem is higher for longer methods than for extremely short methods of one or two statements, but there are usually better ways of achieving the same effect.

Code duplication in general is highly undesirable for a range of reasons. The artificially inflated amount of code is more difficult to understand, and sequences of similar but subtly different lines can mask the real purpose or intention behind them. Also, there is always a risk that only one of several copies of the code is updated to address a defect or add a feature.


## Recommendation
At its simplest, the duplication can be addressed by simply removing all but one of the duplicate method definitions, and changing calls to the removed methods so that they call the remaining function instead.

This may not be possible because of visibility or accessibility. A common example is where two classes implement the same functionality but neither is a subtype of the other, so it is not possible to inherit a single method definition. In such cases, introducing a common superclass to share the duplicated code is a possible option. Alternatively, if the methods do not need access to private object state, they can be moved to a shared utility class that just provides the functionality itself.


## Example
In the following example, `RowWidget` and `ColumnWidget` contain duplicate methods. The `collectChildren` method should probably be moved into the superclass, `Widget`, and shared between `RowWidget` and `ColumnWidget`.


```java
class RowWidget extends Widget {
	// ...
	public void collectChildren(Set<Widget> result) {
		for (Widget child : this.children) {
			if (child.isVisible()) {
				result.add(children);
				child.collectChildren(result);
			}
		}
	}
}

class ColumnWidget extends Widget {
	// ...
	public void collectChildren(Set<Widget> result) {
		for (Widget child : this.children) {
			if (child.isVisible()) {
				result.add(children);
				child.collectChildren(result);
			}
		}
	}
}
```
Alternatively, if not all kinds of `Widget` actually need `collectChildren` (for example, not all of them have children), it might be necessary to introduce a new, possibly abstract, class under `Widget`. For example, the new class might be called `ContainerWidget` and include a single definition of `collectChildren`. Both `RowWidget` and `ColumnWidget` could extend the class and inherit `collectChildren`.

Modern IDEs may provide refactoring support for this sort of issue, usually with the names "Pull up" or "Extract supertype".


## References
* E. Juergens, F. Deissenboeck, B. Hummel, S. Wagner. *Do code clones matter?* Proceedings of the 31st International Conference on Software Engineering, 485-495, 2009.

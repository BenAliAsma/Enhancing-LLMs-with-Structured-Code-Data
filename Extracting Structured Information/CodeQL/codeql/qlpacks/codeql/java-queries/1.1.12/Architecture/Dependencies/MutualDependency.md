# Mutually-dependent types
A mutual dependency exists when two code entities (for example, types or packages) depend directly on each other. Mutual dependencies are caused by unwanted dependencies in one or both directions. There are many different kinds of dependency; here are a few examples of how an inter-type dependency from `T1` to `T2` can occur:

* `T1` derives from a type involving `T2`, for example `T2` itself or `List<T2>`.
* `T1` declares a field of a type involving `T2`.
* `T1` declares a method whose return type involves `T2`.
* A method of `T1` declares a local variable whose type involves `T2`.
* A method of `T1` catches an exception of a type involving `T2`.
Mutual dependencies prevent you from considering either entity in isolation, affecting readability and testability. For example, if types `T1` and `T2` depend on each other, then it is generally impossible to fully understand `T1` without understanding `T2`, and vice-versa. Moreover, neither type can be tested without the other being present. Whilst mocking can alleviate this latter problem to some extent, breaking the mutual dependency is a better solution. For example, suppose we could remove all of the dependencies from `T2` to `T1` - in that case, we would be able to test `T2` in isolation, and completely side-step the need to provide a `T1`, mocked or otherwise.


## Recommendation
Breaking mutual dependencies involves finding ways of removing the unwanted individual dependencies that cause them. The way to do this depends on the kind of dependency in question, with some kinds (for example, dependencies caused by inheritance) being much harder to break than others. A full list of ways to break cycles is beyond the scope of this help topic, however, a few high-level techniques for breaking a dependency from `T1` to `T2` include:

* Introducing an interface that is implemented by `T2`. `T1` can then be refactored to use `T2` only via the interface, which breaks the cycle.
* Moving the depended-on code in `T2` to a third (possibly new) entity. `T1` can then depend on this third entity instead of on `T2`, breaking the cycle. `T2` is allowed to depend on the third entity as well, although it does not have to if there is no need.
* Merging `T1` and `T2` together (for example, if there was an artificial separation between two parts of the same concept). This is not a generally-applicable solution, but is sometimes the right thing to do. It has the effect of internalizing the cycle, which is sufficient to solve the problem.
For more information on how to break unwanted dependencies, see the references (particularly \[Lakos\]).


## Example
In this example `BadModel` and `BadView` are mutually dependent.


```java
public class MutualDependency {
	// Violation: BadModel and BadView are mutually dependent
	private static class BadModel {
		private int i;
		private BadView view;

		public int getI() {
			return i;
		}

		public void setI(int i) {
			this.i = i;
			if(view != null) view.modelChanged();
		}

		public void setView(BadView view) {
			this.view = view;
		}
	}

	private static class BadView {
		private BadModel model;

		public BadView(BadModel model) {
			this.model = model;
		}

		public void modelChanged() {
			System.out.println("Model Changed: " + model.getI());
		}
	}
}
```
The interface technique can be used to break the dependency between the model and the view. The `ModelListener` interface allows `BetterView` to interact with `BetterModel` without dependency.


```java
public class NoMutualDependency {
	// Better: A new interface breaks the dependency
	// from the model to the view
	private interface ModelListener {
		void modelChanged();
	}

	private static class BetterModel {
		private int i;
		private ModelListener listener;

		public int getI() {
			return i;
		}

		public void setI(int i) {
			this.i = i;
			if (listener != null) listener.modelChanged();
		}

		public void setListener(ModelListener listener) {
			this.listener = listener;
		}
	}

	private static class BetterView implements ModelListener {
		private BetterModel model;

		public BetterView(BetterModel model) {
			this.model = model;
		}

		public void modelChanged() {
			System.out.println("Model Changed: " + model.getI());
		}
	}

	public static void main(String[] args) {
		BadModel badModel = new BadModel();
		BadView badView = new BadView(badModel);
		badModel.setView(badView);
		badModel.setI(23);
		badModel.setI(9);

		BetterModel betterModel = new BetterModel();
		BetterView betterView = new BetterView(betterModel);
		betterModel.setListener(betterView);
		betterModel.setI(24);
		betterModel.setI(12);
	}
}
```

## References
* J. Lakos. *Large-Scale C++ Software Design*. Addison-Wesley, 1996.
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.

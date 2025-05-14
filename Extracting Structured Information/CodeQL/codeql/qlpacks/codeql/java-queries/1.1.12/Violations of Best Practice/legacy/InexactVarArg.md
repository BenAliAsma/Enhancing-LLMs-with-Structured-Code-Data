# Inexact type match for varargs argument
A variable arity method, commonly known as a varargs method, may be called with different numbers of arguments. For example, the method `sum(int...&nbsp;values)` may be called in all of the following ways:

* `sum()`
* `sum(1)`
* `sum(1,2,3)`
* `sum(new int[] { 1, 2, 3 })`
When a method `foo(T...&nbsp;x)` is called with an argument that is neither `T` nor `T[]`, but the argument can be cast as either, the choice of which type the argument is cast as is compiler-dependent.


## Recommendation
When a variable arity method, for example `m(T... ts)`, is called with a single argument (for example `m(arg)`), the type of the argument should be either `T` or `T[]` (insert a cast if necessary).


## Example
In the following example, the calls to `length` do not pass an argument of the same type as the parameter of `length`, which is `Object` or an array of `Object`. Therefore, when the program is compiled with javac, the output is:

```java

3
2

```
When the program is compiled with a different compiler, for example the default compiler for some versions of Eclipse, the output may be:

```java

3
1

```

```java
class InexactVarArg
{
	private static void length(Object... objects) {
		System.out.println(objects.length);
	}

	public static void main(String[] args) {
		String[] words = { "apple", "banana", "cherry" };
		String[][] lists = { words, words };
		length(words);	// avoid: Argument does not clarify
		length(lists);	// which parameter type is used.
	}
}

```
To avoid this compiler-dependent behavior, `length(words)` should be replaced by either of the following:

* `length((Object) words)`
* `length((Object[]) words)`
Similarly, `length(lists)` should be replaced by one of the following:

* `length((Object) lists)`
* `length((Object[]) lists)`

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [8.4.1 Formal Parameters](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.4.1), [15.12.4.2 Evaluate Arguments](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.12.4.2).

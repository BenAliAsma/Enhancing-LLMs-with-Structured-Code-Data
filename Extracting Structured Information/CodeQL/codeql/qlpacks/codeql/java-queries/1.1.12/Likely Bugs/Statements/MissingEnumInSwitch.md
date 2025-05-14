# Missing enum case in switch
A `switch` statement that is based on a variable with an `enum` type should either have a default case or handle all possible constants of that `enum` type. Handling all but one or two `enum` constants is usually a coding mistake.


## Recommendation
If there are only a handful of missing cases, add them to the end of the `switch` statement. If there are many cases that do not need to be handled individually, add a default case to handle them.

If there are some `enum` constants that should never occur in this particular part of the code, then program defensively by adding cases for those constants and explicitly throwing an exception (rather than just having no cases for those constants).


## Example
In the following example, the case for 'YES' is missing. Therefore, if `answer` is 'YES', an exception is thrown at run time. To fix this, a case for 'YES' should be added.


```java
enum Answer { YES, NO, MAYBE }

class Optimist
{
	Answer interpret(Answer answer) {
		switch (answer) {
			case MAYBE:
				return Answer.YES;
			case NO:
				return Answer.MAYBE;
			// Missing case for 'YES'
		}
		throw new RuntimeException("uncaught case: " + answer);
	}
}

```

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [8.9 Enum Types](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.9), [14.11 The switch Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.11).
* Common Weakness Enumeration: [CWE-478](https://cwe.mitre.org/data/definitions/478.html).

# Assignment to parameter
Programmers usually assume that the value of a parameter is the value that was passed in to the method or constructor. Assigning a different value to a parameter in a method or constructor invalidates that assumption.


## Recommendation
Avoid assignment to parameters by doing one of the following:

* Introduce a local variable and assign to that instead.
* Use an expression directly rather than assigning it to a parameter.

## Example
In the following example, the first method shows assignment to the parameter `miles`. The second method shows how to avoid this by using the expression `miles * KM_PER_MILE`. The third method shows how to avoid the assignment by declaring a local variable `kilometres` and assigning to that.


```java
final private static double KM_PER_MILE = 1.609344;

// AVOID: Example that assigns to a parameter
public double milesToKM(double miles) {
	miles *= KM_PER_MILE;
	return miles;
}

// GOOD: Example of using an expression instead
public double milesToKM(double miles) {
	return miles * KM_PER_MILE;
}

// GOOD: Example of using a local variable
public double milesToKM(double miles) {
	double kilometres = miles * KM_PER_MILE;
	return kilometres;
}

```

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Basics: [Methods 4 - Local variables](https://web.archive.org/web/20200223080939/http://leepoint.net/JavaBasics/methods/methods-22-local-variables.html).

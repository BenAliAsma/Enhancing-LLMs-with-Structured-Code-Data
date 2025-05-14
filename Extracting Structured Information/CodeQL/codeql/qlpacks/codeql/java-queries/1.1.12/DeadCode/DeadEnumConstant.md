# Dead enum constant
Enum constants that are never used at runtime are redundant and should be removed.

Redundant, or "dead", code imposes a burden on those reading or maintaining the software project. It can make it harder to understand the structure of the code, as well as increasing the complexity of adding new features or fixing bugs. It can also affect compilation and build times for the project, as dead code will still be compiled and built even if it is never used. In some cases it may also affect runtime performance - for example, fields that are written to but never read from, where the value written to the field is expensive to compute. Removing dead code should not change the meaning of the program.

An enum constant is considered dead if at runtime it is never used, or only used in comparisons. Any enum constant which is not dead is considered to be "live".

An enum constant that is only used in a comparison is considered dead because the comparison will always produce the same result. This is because no variable holds the value of the enum constant, so the comparison of any variable against the constant will always return the same result.

A class, method, or field may be dead even if it has dependencies from other parts of the program, if those dependencies are from code that is also considered to be dead. We can also consider this from the opposite side - an element is live, if and only if there is an entry point - such as a `main` method - that eventually calls the method, reads the field or constructs the class.

When identifying dead code, we make an assumption that the snapshot of the project includes all possible callers of the code. If the project is a library project, this may not be the case, and code may be flagged as dead when it is only used by other projects not included in the snapshot.

You can customize the results by defining additional "entry points" or by identifying fields that are accessed using reflection. You may also wish to "whitelist" classes, methods or fields that should be excluded from the results. Please refer to the Semmle documentation for more information.


## Recommendation
Before making any changes, confirm that the enum constant is not required by verifying that the enum constant is never used. This confirmation is necessary because there may be project-specific frameworks or techniques which can introduce hidden dependencies. If this project is for a library, then consider whether the enum constant is part of the external API, and may be used in external projects that are not included in the snapshot.

After confirming that the enum constant is not required, remove the enum constant. You will also need to remove any references to this enum constant, which may, in turn, require removing other dead code.

If you observe a large number of false positives, you may need to add extra entry points to identify hidden dependencies caused by the use of a particular framework or technique, or to identify library project entry points. Please refer to the Semmle documentation for more information on how to do this.


## Example
In the following example, we have an enum class called `Result`, intended to report the result of some operation:


```java
public enum Result {
	SUCCESS,
	FAILURE,
	ERROR
}

public Result runOperation(String value) {
	if (value == 1) {
		return SUCCESS;
	} else {
		return FAILURE;
	}
}

public static void main(String[] args) {
	Result operationResult = runOperation(args[0]);
	if (operationResult == Result.ERROR) {
		exit(1);
	} else {
		exit(0);
	}

}
```
The method `runOperation` performs some operation, and returns a `Result` depending on whether the operation succeeded. However, it only returns either `SUCCESS` or `FAILURE`, and never `ERROR`. The `main` method calls `runOperation`, and checks whether the returned result is the `ERROR`. However, this check will always return the same result - `false`. This is because the `operationResult` can never hold `ERROR`, because `ERROR` is never stored or returned anywhere in the program. Therefore, `ERROR` is dead and can be removed, along with the comparison check, and the `exit(1);`.


## References
* Wikipedia: [Redundant code](https://en.wikipedia.org/wiki/Redundant_code).
* CERT Java Coding Standard: [MSC56-J. Detect and remove superfluous code and values](https://www.securecoding.cert.org/confluence/display/java/MSC56-J.+Detect+and+remove+superfluous+code+and+values).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).

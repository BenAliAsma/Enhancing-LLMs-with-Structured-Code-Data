# Information exposure through a stack trace
Software developers often add stack traces to error messages, as a debugging aid. Whenever that error message occurs for an end user, the developer can use the stack trace to help identify how to fix the problem. In particular, stack traces can tell the developer more about the sequence of events that led to a failure, as opposed to merely the final state of the software when the error occurred.

Unfortunately, the same information can be useful to an attacker. The sequence of class names in a stack trace can reveal the structure of the application as well as any internal components it relies on.


## Recommendation
Send the user a more generic error message that reveals less information. Either suppress the stack trace entirely, or log it only on the server.


## Example
In the following example, an exception is handled in two different ways. In the first version, labeled BAD, the exception is sent back to the remote user using the `sendError()` method. As such, the user is able to see a detailed stack trace, which may contain sensitive information. In the second version, the error message is logged only on the server. That way, the developers can still access and use the error log, but remote users will not see the information.


```java
protected void doGet(HttpServletRequest request, HttpServletResponse response) {
	try {
		doSomeWork();
	} catch (NullPointerException ex) {
		// BAD: printing a stack trace back to the response
		ex.printStackTrace(response.getWriter());
		return;
	}

	try {
		doSomeWork();
	} catch (NullPointerException ex) {
		// GOOD: log the stack trace, and send back a non-revealing response
		log("Exception occurred", ex);
		response.sendError(
			HttpServletResponse.SC_INTERNAL_SERVER_ERROR,
			"Exception occurred");
		return;
	}
}

```

## References
* OWASP: [Improper Error Handling](https://owasp.org/www-community/Improper_Error_Handling).
* CERT Java Coding Standard: [ERR01-J. Do not allow exceptions to expose sensitive information](https://www.securecoding.cert.org/confluence/display/java/ERR01-J.+Do+not+allow+exceptions+to+expose+sensitive+information).
* Common Weakness Enumeration: [CWE-209](https://cwe.mitre.org/data/definitions/209.html).
* Common Weakness Enumeration: [CWE-497](https://cwe.mitre.org/data/definitions/497.html).

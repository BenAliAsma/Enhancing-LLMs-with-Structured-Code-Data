# Information exposure through an error message
The error message at the top of a stack trace can include information such as server-side file names and SQL code that the application relies on, allowing an attacker to fine-tune a subsequent injection attack.


## Recommendation
Send the user a more generic error message that reveals less information. Either suppress the error message entirely, or log it only on the server.


## Example
In the following example, an exception is handled in two different ways. In the first version, labeled BAD, the exception is sent back to the remote user using the `getMessage()` method. As such, the user is able to see a detailed error message, which may contain sensitive information. In the second version, the error message is logged only on the server. That way, the developers can still access and use the error log, but remote users will not see the information.


```java
protected void doGet(HttpServletRequest request, HttpServletResponse response) {
	try {
		doSomeWork();
	} catch (NullPointerException ex) {
		// BAD: printing a exception message back to the response
		response.sendError(
			HttpServletResponse.SC_INTERNAL_SERVER_ERROR,
			ex.getMessage());
		return;
	}

	try {
		doSomeWork();
	} catch (NullPointerException ex) {
		// GOOD: log the exception message, and send back a non-revealing response
		log("Exception occurred", ex.getMessage);
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

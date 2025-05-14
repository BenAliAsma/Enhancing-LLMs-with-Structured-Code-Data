# User-controlled bypass of sensitive method
Many Java constructs enable code statements to be executed conditionally, for example `if` statements and `for` statements. If these statements contain important authentication or login code, and the decision about whether to execute this code is based on user-controlled data, it may be possible for an attacker to bypass security systems by preventing this code from executing.


## Recommendation
Never decide whether to authenticate a user based on data that may be controlled by that user. If necessary, ensure that the data is validated extensively when it is input before any authentication checks are performed.

It is still possible to have a system that "remembers" users, thus not requiring the user to login on every interaction. For example, personalization settings can be applied without authentication because this is not sensitive information. However, users should be allowed to take sensitive actions only when they have been fully authenticated.


## Example
This example shows two ways of deciding whether to authenticate a user. The first way shows a decision that is based on the value of a cookie. Cookies can be easily controlled by the user, and so this allows a user to become authenticated without providing valid credentials. The second, more secure way shows a decision that is based on looking up the user in a security database.


```java
public boolean doLogin(String user, String password) {
	Cookie adminCookie = getCookies()[0];

	// BAD: login is executed only if the value of 'adminCookie' is 'false', 
	// but 'adminCookie' is controlled by the user
	if(adminCookie.getValue()=="false")
		return login(user, password);
	
	return true;
}

public boolean doLogin(String user, String password) {
	Cookie adminCookie = getCookies()[0];
	
	// GOOD: use server-side information based on the credentials to decide
	// whether user has privileges
	boolean isAdmin = queryDbForAdminStatus(user, password);
	if(!isAdmin)
		return login(user, password);
	
	return true;
}
```

## References
* SEI CERT Oracle Coding Standard for Java: [SEC02-J. Do not base security checks on untrusted sources](https://wiki.sei.cmu.edu/confluence/display/java/SEC02-J.+Do+not+base+security+checks+on+untrusted+sources).
* Common Weakness Enumeration: [CWE-807](https://cwe.mitre.org/data/definitions/807.html).
* Common Weakness Enumeration: [CWE-290](https://cwe.mitre.org/data/definitions/290.html).

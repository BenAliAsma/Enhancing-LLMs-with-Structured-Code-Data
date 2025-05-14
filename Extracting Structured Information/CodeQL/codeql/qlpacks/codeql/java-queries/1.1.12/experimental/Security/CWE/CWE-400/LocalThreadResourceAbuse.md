# Uncontrolled thread resource consumption from local input source
The `Thread.sleep` method is used to pause the execution of current thread for specified time. When the sleep time is user-controlled, especially in the web application context, it can be abused to cause all of a server's threads to sleep, leading to denial of service.


## Recommendation
To guard against this attack, consider specifying an upper range of allowed sleep time or adopting the producer/consumer design pattern with `Object.wait` method to avoid performance problems or even resource exhaustion. For more information, refer to the concurrency tutorial of Oracle listed below or `java/ql/src/Likely Bugs/Concurrency` queries of CodeQL.


## Example
The following example shows a bad situation and a good situation respectively. In the bad situation, a thread sleep time comes directly from user input. In the good situation, an upper range check on the maximum sleep time allowed is enforced.


```java
class SleepTest {
	public void test(int userSuppliedWaitTime) throws Exception {
		// BAD: no boundary check on wait time
		Thread.sleep(userSuppliedWaitTime);

		// GOOD: enforce an upper limit on wait time
		if (userSuppliedWaitTime > 0 && userSuppliedWaitTime < 5000) {
			Thread.sleep(userSuppliedWaitTime);
		}
	}
}

```

## References
* Snyk: [Denial of Service (DoS) in com.googlecode.gwtupload:gwtupload](https://snyk.io/vuln/SNYK-JAVA-COMGOOGLECODEGWTUPLOAD-569506).
* gwtupload: [\[Fix DOS issue\] Updating the AbstractUploadListener.java file](https://github.com/manolo/gwtupload/issues/33).
* The blog of a gypsy engineer: [ CVE-2019-17555: DoS via Retry-After header in Apache Olingo](https://blog.gypsyengineer.com/en/security/cve-2019-17555-dos-via-retry-after-header-in-apache-olingo.html).
* Oracle: [The Java Concurrency Tutorials](https://docs.oracle.com/javase/tutorial/essential/concurrency/guardmeth.html)
* Common Weakness Enumeration: [CWE-400](https://cwe.mitre.org/data/definitions/400.html).

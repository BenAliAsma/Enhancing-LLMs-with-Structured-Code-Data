# Insertion of sensitive information into log files
Information written to log files can be of a sensitive nature and give valuable guidance to an attacker or expose sensitive user information. Third-party logging utilities like Log4J and SLF4J are widely used in Java projects. When sensitive information is written to logs without properly set logging levels, it is accessible to potential attackers who can use it to gain access to file storage.


## Recommendation
Do not write secrets into the log files and enforce proper logging level control.


## Example
The following example shows two ways of logging sensitive information. In the 'BAD' case, the credentials are simply written to a debug log. In the 'GOOD' case, the credentials are never written to debug logs.


```java
public static void main(String[] args) {
    {
        private static final Logger logger = LogManager.getLogger(SensitiveInfoLog.class);

        String password = "Pass@0rd";

        // BAD: user password is written to debug log
        logger.debug("User password is "+password);
    }
	
    {
        private static final Logger logger = LogManager.getLogger(SensitiveInfoLog.class);
  
        String password = "Pass@0rd";

        // GOOD: user password is never written to debug log
        logger.debug("User password changed")
    }
}

```

## References
* [OWASP Logging Guide](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)
* Common Weakness Enumeration: [CWE-532](https://cwe.mitre.org/data/definitions/532.html).

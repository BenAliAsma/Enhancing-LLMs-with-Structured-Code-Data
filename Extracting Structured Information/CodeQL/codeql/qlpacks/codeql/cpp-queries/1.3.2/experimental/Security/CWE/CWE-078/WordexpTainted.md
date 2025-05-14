# Uncontrolled data used in `wordexp` command
The code passes user input to `wordexp`. This leaves the code vulnerable to attack by command injection, because `wordexp` performs command substitution. Command substitution is a feature that replaces `$(command)` or ``` `command` ``` with the output of the given command, allowing the user to run arbitrary code on the system.


## Recommendation
When calling `wordexp`, pass the `WRDE_NOCMD` flag to prevent command substitution.


## Example
The following example passes a user-supplied file path to `wordexp` in two ways. The first way uses `wordexp` with no specified flags. As such, it is vulnerable to command injection. The second way uses `wordexp` with the `WRDE_NOCMD` flag. As such, no command substitution is performed, making this safe from command injection.


```c

int main(int argc, char** argv) {
  char *filePath = argv[2];

  {
    // BAD: the user-controlled string is injected
    // directly into `wordexp` which performs command substitution

    wordexp_t we;
    wordexp(filePath, &we, 0);
  }

  {
    // GOOD: command substitution is disabled

    wordexp_t we;
    wordexp(filePath, &we, WRDE_NOCMD);
  }
}

```

## References
* CERT C Coding Standard: [STR02-C. Sanitize data passed to complex subsystems](https://www.securecoding.cert.org/confluence/display/c/STR02-C.+Sanitize+data+passed+to+complex+subsystems).
* OWASP: [Command Injection](https://www.owasp.org/index.php/Command_Injection).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).

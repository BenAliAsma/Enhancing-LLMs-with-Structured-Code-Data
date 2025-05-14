# Uncontrolled data used in OS command
The code passes user input as part of a call to `system` or `popen` without escaping special elements. It generates a command line using `sprintf`, with the user-supplied data directly passed as a formatting argument. This leaves the code vulnerable to attack by command injection.


## Recommendation
Use a library routine to escape characters in the user-supplied string before passing it to a command shell.


## Example
The following example runs an external command in two ways. The first way uses `sprintf` to build a command directly out of a user-supplied argument. As such, it is vulnerable to command injection. The second way quotes the user-provided value before embedding it in the command; assuming the `encodeShellString` utility is correct, this code should be safe against command injection.


```c
int main(int argc, char** argv) {
  char *userName = argv[2];
  
  {
    // BAD: a string from the user is injected directly into
    // a command line.
    char command1[1000] = {0};
    sprintf(command1, "userinfo -v \"%s\"", userName);
    system(command1);
  }

  {
    // GOOD: the user string is encoded by a library routine.
    char userNameQuoted[1000] = {0};
    encodeShellString(userNameQuoted, 1000, userName); 
    char command2[1000] = {0};
    sprintf(command2, "userinfo -v %s", userNameQuoted);
    system(command2);
  }
}

```

## References
* CERT C Coding Standard: [STR02-C. Sanitize data passed to complex subsystems](https://www.securecoding.cert.org/confluence/display/c/STR02-C.+Sanitize+data+passed+to+complex+subsystems).
* OWASP: [Command Injection](https://www.owasp.org/index.php/Command_Injection).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).
* Common Weakness Enumeration: [CWE-88](https://cwe.mitre.org/data/definitions/88.html).

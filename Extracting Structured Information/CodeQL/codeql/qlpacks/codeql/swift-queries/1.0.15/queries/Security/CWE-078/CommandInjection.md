# System command built from user-controlled sources
Constructing a system command with unsanitized user input is dangerous, since a malicious user may be able to craft input that executes arbitrary code.


## Recommendation
If possible, use hard-coded string literals to specify the command to run. Instead of interpreting user input directly as command names, examine the input and then choose among hard-coded string literals.

If this is not possible, then add sanitization code to verify that the user input is safe before using it.


## Example
The following example executes code from user input without sanitizing it first:


```swift
var task = Process()
task.launchPath = "/bin/bash"
task.arguments = ["-c", userControlledString] // BAD

task.launch()
```
If user input is used to construct a command it should be checked first. This ensures that the user cannot insert characters that have special meanings:


```swift
func validateCommand(_ command: String) -> String? {
    let allowedCommands = ["ls -l", "pwd", "echo"]
    if allowedCommands.contains(command) {
        return command
    }
    return nil
}

if let validatedString = validateCommand(userControlledString) {
    var task = Process()
    task.launchPath = "/bin/bash"
    task.arguments = ["-c", validatedString] // GOOD

    task.launch()
}

```

## References
* OWASP: [Command Injection](https://www.owasp.org/index.php/Command_Injection).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).
* Common Weakness Enumeration: [CWE-88](https://cwe.mitre.org/data/definitions/88.html).

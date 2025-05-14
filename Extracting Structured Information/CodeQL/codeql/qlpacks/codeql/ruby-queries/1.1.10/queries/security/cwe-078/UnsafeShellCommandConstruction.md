# Unsafe shell command constructed from library input
Dynamically constructing a shell command with inputs from exported functions may inadvertently change the meaning of the shell command. Clients using the exported function may use inputs containing characters that the shell interprets in a special way, for instance quotes and spaces. This can result in the shell command misbehaving, or even allowing a malicious user to execute arbitrary commands on the system.


## Recommendation
If possible, avoid concatenating shell strings to APIs such as `system(..)` to avoid interpretation by the shell.

Instead, provide the arguments to the shell command as separate arguments to the API, such as `system("echo", arg1, arg2)`.

Alternatively, if the shell command must be constructed dynamically, then add code to ensure that special characters do not alter the shell command unexpectedly.


## Example
The following example shows a dynamically constructed shell command that downloads a file from a remote URL.


```ruby
module Utils 
    def download(path)
        system("wget #{path}") # NOT OK
    end
end
```
The shell command will, however, fail to work as intended if the input contains spaces or other special characters interpreted in a special way by the shell.

Even worse, a client might pass in user-controlled data, not knowing that the input is interpreted as a shell command. This could allow a malicious user to provide the input `http://example.org; cat /etc/passwd` in order to execute the command `cat /etc/passwd`.

To avoid such potentially catastrophic behaviors, provide the input from exported functions as an argument that does not get interpreted by a shell:


```ruby
module Utils 
    def download(path)
        # using an API that doesn't interpret the path as a shell command
        system("wget", path) # OK
    end
end
```

## References
* OWASP: [Command Injection](https://www.owasp.org/index.php/Command_Injection).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).
* Common Weakness Enumeration: [CWE-88](https://cwe.mitre.org/data/definitions/88.html).
* Common Weakness Enumeration: [CWE-73](https://cwe.mitre.org/data/definitions/73.html).

# Unsafe shell command constructed from library input
Dynamically constructing a shell command with inputs from exported functions may inadvertently change the meaning of the shell command. Clients using the exported function may use inputs containing characters that the shell interprets in a special way, for instance quotes and spaces. This can result in the shell command misbehaving, or even allowing a malicious user to execute arbitrary commands on the system.


## Recommendation
If possible, provide the dynamic arguments to the shell as an array using a safe API such as `child_process.execFile` to avoid interpretation by the shell.

If given arguments as a single string, avoid simply splitting the string on whitespace. Arguments may contain quoted whitespace, causing them to split into multiple arguments. Use a library like `shell-quote` to parse the string into an array of arguments instead.

Alternatively, if the command must be interpreted by a shell (for example because it includes I/O redirections), you can use `shell-quote` to escape any special characters in the input before embedding it in the command.


## Example
The following example shows a dynamically constructed shell command that downloads a file from a remote URL.


```javascript
var cp = require("child_process");

module.exports = function download(path, callback) {
  cp.exec("wget " + path, callback);
}

```
The shell command will, however, fail to work as intended if the input contains spaces or other special characters interpreted in a special way by the shell.

Even worse, a client might pass in user-controlled data, not knowing that the input is interpreted as a shell command. This could allow a malicious user to provide the input `http://example.org; cat /etc/passwd` in order to execute the command `cat /etc/passwd`.

To avoid such potentially catastrophic behaviors, provide the inputs from exported functions as an argument that does not get interpreted by a shell:


```javascript
var cp = require("child_process");

module.exports = function download(path, callback) {
  cp.execFile("wget", [path], callback);
}

```
As another example, consider the following code which is similar to the preceding example, but pipes the output of `wget` into `wc -l` to count the number of lines in the downloaded file.


```javascript
var cp = require("child_process");

module.exports = function download(path, callback) {
  cp.exec("wget " + path + " | wc -l", callback);
};

```
In this case, using `child_process.execFile` is not an option because the shell is needed to interpret the pipe operator. Instead, you can use `shell-quote` to escape the input before embedding it in the command:


```javascript
var cp = require("child_process");

module.exports = function download(path, callback) {
  cp.exec("wget " + shellQuote.quote([path]) + " | wc -l", callback);
};

```

## References
* OWASP: [Command Injection](https://www.owasp.org/index.php/Command_Injection).
* npm: [shell-quote](https://www.npmjs.com/package/shell-quote).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).
* Common Weakness Enumeration: [CWE-88](https://cwe.mitre.org/data/definitions/88.html).

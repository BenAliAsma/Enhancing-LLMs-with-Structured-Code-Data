# Indirect uncontrolled command line
Forwarding command-line arguments to `child_process.exec` or some other library routine that executes a system command within a shell can change the meaning of the command unexpectedly due to unescaped special characters.

When the forwarded command-line arguments come from a parent process that has not escaped the special characters in the arguments, then the parent process may indirectly be vulnerable to command-line injection since the special characters are evaluated unexpectedly.


## Recommendation
If possible, use APIs that don't run shell commands and accept command arguments as an array of strings rather than a single concatenated string. This is both safer and more portable.

If given arguments as a single string, avoid simply splitting the string on whitespace. Arguments may contain quoted whitespace, causing them to split into multiple arguments. Use a library like `shell-quote` to parse the string into an array of arguments instead.

If this approach is not viable, then add code to verify that each forwarded command-line argument is properly escaped before using it.


## Example
The following wrapper script example executes another JavaScript file in a child process and forwards some command-line arguments. This is problematic because the special characters in the command-line arguments may change the meaning of the child process invocation unexpectedly. For instance, if one of the command-line arguments is `"dollar$separated$name"`, then the child process will substitute the two environment variables `$separated` and `$name` before invoking `node`.


```javascript
var cp = require("child_process");

const args = process.argv.slice(2);
const script = path.join(__dirname, 'bin', 'main.js');
cp.execSync(`node ${script} ${args.join(' ')}`); // BAD

```
If another program uses `child_process.execFile` to invoke the above wrapper script with input from a remote user, then there may be a command-line injection vulnerability. This may be surprising, since a command-line invocation with `child_process.execFile` is generally considered safe. But in this case, the remote user input is simply forwarded to the problematic `process.exec` call in the wrapper script.

To guard against this, use an API that does not perform environment variable substitution, such as `child_process.execFile`:


```javascript
var cp = require("child_process");

const args = process.argv.slice(2);
const script = path.join(__dirname, 'bin', 'main.js');
cp.execFileSync('node', [script].concat(args)); // GOOD

```
If you want to allow the user to specify other options to `node`, you can use a library like `shell-quote` to parse the user input into an array of arguments without risking command injection:


```javascript
var cp = require("child_process"),
    shellQuote = require("shell-quote");

const args = process.argv.slice(2);
let nodeOpts = '';
if (args[0] === '--node-opts') {
    nodeOpts = args[1];
    args.splice(0, 2);
}
const script = path.join(__dirname, 'bin', 'main.js');
cp.execFileSync('node', shellQuote.parse(nodeOpts).concat(script).concat(args)); // GOOD

```

## References
* OWASP: [Command Injection](https://www.owasp.org/index.php/Command_Injection).
* npm: [shell-quote](https://www.npmjs.com/package/shell-quote).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).
* Common Weakness Enumeration: [CWE-88](https://cwe.mitre.org/data/definitions/88.html).

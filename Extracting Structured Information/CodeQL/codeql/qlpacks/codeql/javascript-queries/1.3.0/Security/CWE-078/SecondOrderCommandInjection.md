# Second order command injection
Some shell commands, like `git ls-remote`, can execute arbitrary commands if a user provides a malicious URL that starts with `--upload-pack`. This can be used to execute arbitrary code on the server.


## Recommendation
Sanitize user input before passing it to the shell command. For example, ensure that URLs are valid and do not contain malicious commands.


## Example
The following example shows code that executes `git ls-remote` on a URL that can be controlled by a malicious user.


```javascript
const express = require("express");
const app = express();

const cp = require("child_process");

app.get("/ls-remote", (req, res) => {
  const remote = req.query.remote;
  cp.execFile("git", ["ls-remote", remote]); // NOT OK
});

```
The problem has been fixed in the snippet below, where the URL is validated before being passed to the shell command.


```javascript
const express = require("express");
const app = express();

const cp = require("child_process");

app.get("/ls-remote", (req, res) => {
  const remote = req.query.remote;
  if (!(remote.startsWith("git@") || remote.startsWith("https://"))) {
    throw new Error("Invalid remote: " + remote);
  }
  cp.execFile("git", ["ls-remote", remote]); // OK
});

```

## References
* Max Justicz: [Hacking 3,000,000 apps at once through CocoaPods](https://justi.cz/security/2021/04/20/cocoapods-rce.html).
* Git: [Git - git-ls-remote Documentation](https://git-scm.com/docs/git-ls-remote/2.22.0#Documentation/git-ls-remote.txt---upload-packltexecgt).
* OWASP: [Command Injection](https://www.owasp.org/index.php/Command_Injection).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).
* Common Weakness Enumeration: [CWE-88](https://cwe.mitre.org/data/definitions/88.html).

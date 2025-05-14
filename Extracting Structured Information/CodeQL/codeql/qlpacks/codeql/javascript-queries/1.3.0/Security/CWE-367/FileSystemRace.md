# Potential file system race condition
Often it is necessary to check the state of a file before using it. These checks usually take a file name to be checked, and if the check returns positively, then the file is opened or otherwise operated upon.

However, in the time between the check and the operation, the underlying file referenced by the file name could be changed by an attacker, causing unexpected behavior.


## Recommendation
Use file descriptors instead of file names whenever possible.


## Example
The following example shows a case where the code checks whether a file inside the `/tmp/` folder exists, and if it doesn't, the file is written to that location.


```javascript
const fs = require("fs");
const os = require("os");
const path = require("path");

const filePath = path.join(os.tmpdir(), "my-temp-file.txt");

if (!fs.existsSync(filePath)) {
  fs.writeFileSync(filePath, "Hello", { mode: 0o600 });
}

```
However, in a multi-user environment the file might be created by another user between the existence check and the write.

This can be avoided by using `fs.open` to get a file descriptor, and then use that file descriptor in the write operation.


```javascript
const fs = require("fs");
const os = require("os");
const path = require("path");

const filePath = path.join(os.tmpdir(), "my-temp-file.txt");

try {
  const fd = fs.openSync(filePath, fs.O_CREAT | fs.O_EXCL | fs.O_RDWR, 0o600);

  fs.writeFileSync(fd, "Hello");
} catch (e) {
  // file existed
}

```

## References
* Wikipedia: [Time-of-check to time-of-use](https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use).
* The CERT Oracle Secure Coding Standard for C: [ FIO01-C. Be careful using functions that use file names for identification ](https://www.securecoding.cert.org/confluence/display/c/FIO01-C.+Be+careful+using+functions+that+use+file+names+for+identification).
* NodeJS: [The FS module](https://nodejs.org/api/fs.html).
* Common Weakness Enumeration: [CWE-367](https://cwe.mitre.org/data/definitions/367.html).

# Uncontrolled data used in path expression
Accessing files using paths constructed from user-controlled data can allow an attacker to access unexpected resources. This can result in sensitive information being revealed or deleted, or an attacker being able to influence behavior by modifying unexpected files.


## Recommendation
Validate user input before using it to construct a file path.

The validation method you should use depends on whether you want to allow the user to specify complex paths with multiple components that may span multiple folders, or only simple filenames without a path component.

In the former case, a common strategy is to make sure that the constructed file path is contained within a safe root folder. First, normalize the path using `path.resolve` or `fs.realpathSync` to remove any ".." segments. You should always normalize the file path since an unnormalized path that starts with the root folder can still be used to access files outside the root folder. Then, after you have normalized the path, check that the path starts with the root folder.

In the latter case, you can use a library like the `sanitize-filename` npm package to eliminate any special characters from the file path. Note that it is *not* sufficient to only remove "../" sequences: for example, applying this filter to ".../...//" would still result in the string "../".

Finally, the simplest (but most restrictive) option is to use an allow list of safe patterns and make sure that the user input matches one of these patterns.


## Example
In the first (bad) example, the code reads the file name from an HTTP request, then accesses that file within a root folder. A malicious user could enter a file name containing "../" segments to navigate outside the root folder and access sensitive files.


```javascript
const fs = require('fs'),
      http = require('http'),
      url = require('url');

const ROOT = "/var/www/";

var server = http.createServer(function(req, res) {
  let filePath = url.parse(req.url, true).query.path;

  // BAD: This function uses unsanitized input that can read any file on the file system.
  res.write(fs.readFileSync(ROOT + filePath, 'utf8'));
});
```
The second (good) example shows how to avoid access to sensitive files by sanitizing the file path. First, the code resolves the file name relative to a root folder, normalizing the path and removing any "../" segments in the process. Then, the code calls `fs.realpathSync` to resolve any symbolic links in the path. Finally, the code checks that the normalized path starts with the path of the root folder, ensuring the file is contained within the root folder.


```javascript
const fs = require('fs'),
      http = require('http'),
      path = require('path'),
      url = require('url');

const ROOT = "/var/www/";

var server = http.createServer(function(req, res) {
  let filePath = url.parse(req.url, true).query.path;

  // GOOD: Verify that the file path is under the root directory
  filePath = fs.realpathSync(path.resolve(ROOT, filePath));
  if (!filePath.startsWith(ROOT)) {
    res.statusCode = 403;
    res.end();
    return;
  }
  res.write(fs.readFileSync(filePath, 'utf8'));
});
```

## References
* OWASP: [Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* npm: [sanitize-filename](https://www.npmjs.com/package/sanitize-filename) package.
* Common Weakness Enumeration: [CWE-22](https://cwe.mitre.org/data/definitions/22.html).
* Common Weakness Enumeration: [CWE-23](https://cwe.mitre.org/data/definitions/23.html).
* Common Weakness Enumeration: [CWE-36](https://cwe.mitre.org/data/definitions/36.html).
* Common Weakness Enumeration: [CWE-73](https://cwe.mitre.org/data/definitions/73.html).
* Common Weakness Enumeration: [CWE-99](https://cwe.mitre.org/data/definitions/99.html).

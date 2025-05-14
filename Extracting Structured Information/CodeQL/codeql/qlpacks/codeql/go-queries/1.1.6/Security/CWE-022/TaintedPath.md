# Uncontrolled data used in path expression
Accessing files using paths constructed from user-controlled data can allow an attacker to access unexpected resources. This can result in sensitive information being revealed or deleted, or an attacker being able to influence behavior by modifying unexpected files.

Paths that are naively constructed from data controlled by a user may be absolute paths, or may contain unexpected special characters such as "..". Such a path could point anywhere on the file system.


## Recommendation
Validate user input before using it to construct a file path.

Common validation methods include checking that the normalized path is relative and does not contain any ".." components, or checking that the path is contained within a safe folder. The method you should use depends on how the path is used in the application, and whether the path should be a single path component.

If the path should be a single path component (such as a file name), you can check for the existence of any path separators ("/" or "\\"), or ".." sequences in the input, and reject the input if any are found.

Note that removing "../" sequences is *not* sufficient, since the input could still contain a path separator followed by "..". For example, the input ".../...//" would still result in the string "../" if only "../" sequences are removed.

Finally, the simplest (but most restrictive) option is to use an allow list of safe patterns and make sure that the user input matches one of these patterns.


## Example
In the first example, a file name is read from an HTTP request and then used to access a file. However, a malicious user could enter a file name which is an absolute path, such as "/etc/passwd".

In the second example, it appears that the user is restricted to opening a file within the `"user"` home directory. However, a malicious user could enter a file name containing special characters. For example, the string `"../../etc/passwd"` will result in the code reading the file located at "/home/user/../../etc/passwd", which is the system's password file. This file would then be sent back to the user, giving them access to password information.


```go
package main

import (
	"io/ioutil"
	"net/http"
	"path/filepath"
)

func handler(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query()["path"][0]

	// BAD: This could read any file on the file system
	data, _ := ioutil.ReadFile(path)
	w.Write(data)

	// BAD: This could still read any file on the file system
	data, _ = ioutil.ReadFile(filepath.Join("/home/user/", path))
	w.Write(data)
}

```
If the input should only be a file name, you can check that it doesn't contain any path separators or ".." sequences.


```go
package main

import (
	"io/ioutil"
	"net/http"
	"path/filepath"
	"strings"
)

func handler(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query()["path"][0]

	// GOOD: ensure that the filename has no path separators or parent directory references
	// (Note that this is only suitable if `path` is expected to have a single component!)
	if strings.Contains(path, "/") || strings.Contains(path, "\\") || strings.Contains(path, "..") {
		http.Error(w, "Invalid file name", http.StatusBadRequest)
		return
	}
	data, _ := ioutil.ReadFile(filepath.Join("/home/user/", path))
	w.Write(data)
}

```
Note that this approach is only suitable if the input is expected to be a single file name.

If the input can be a path with multiple components, you can make it safe by verifying that the path is within a specific directory that is considered safe. You can do this by resolving the input with respect to that directory, and then checking that the resulting path is still within it.


```go
package main

import (
	"io/ioutil"
	"net/http"
	"path/filepath"
	"strings"
)

const safeDir = "/home/user/"

func handler(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query()["path"][0]

	// GOOD: ensure that the resolved path is within the safe directory
	absPath, err := filepath.Abs(filepath.Join(safeDir, path))
	if err != nil || !strings.HasPrefix(absPath, safeDir) {
		http.Error(w, "Invalid file name", http.StatusBadRequest)
		return
	}
	data, _ := ioutil.ReadFile(absPath)
	w.Write(data)
}

```
Note that `/home/user` is just an example, you should replace it with the actual safe directory in your application. Also, while in this example the path of the safe directory is absolute, this may not always be the case, and you may need to resolve it first before checking the input.


## References
* OWASP: [Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* Common Weakness Enumeration: [CWE-22](https://cwe.mitre.org/data/definitions/22.html).
* Common Weakness Enumeration: [CWE-23](https://cwe.mitre.org/data/definitions/23.html).
* Common Weakness Enumeration: [CWE-36](https://cwe.mitre.org/data/definitions/36.html).
* Common Weakness Enumeration: [CWE-73](https://cwe.mitre.org/data/definitions/73.html).
* Common Weakness Enumeration: [CWE-99](https://cwe.mitre.org/data/definitions/99.html).

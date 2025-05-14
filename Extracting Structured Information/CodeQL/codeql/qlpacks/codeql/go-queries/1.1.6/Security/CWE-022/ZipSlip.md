# Arbitrary file access during archive extraction ("Zip Slip")
Extracting files from a malicious zip file, or similar type of archive, is at risk of directory traversal attacks if filenames from the archive are not properly validated. archive paths.

Zip archives contain archive entries representing each file in the archive. These entries include a file path for the entry, but these file paths are not restricted and may contain unexpected special elements such as the directory traversal element (`..`). If these file paths are used to create a filesystem path, then a file operation may happen in an unexpected location. This can result in sensitive information being revealed or deleted, or an attacker being able to influence behavior by modifying unexpected files.

For example, if a zip file contains a file entry `..\sneaky-file`, and the zip file is extracted to the directory `c:\output`, then naively combining the paths would result in an output file path of `c:\output\..\sneaky-file`, which would cause the file to be written to `c:\sneaky-file`.


## Recommendation
Ensure that output paths constructed from zip archive entries are validated to prevent writing files to unexpected locations.

The recommended way of writing an output file from a zip archive entry is to check that "`..`" does not occur in the path.


## Example
In this example an archive is extracted without validating file paths. If `archive.zip` contained relative paths (for instance, if it were created by something like `zip archive.zip ../file.txt`) then executing this code could write to locations outside the destination directory.


```go
package main

import (
	"archive/zip"
	"io/ioutil"
	"path/filepath"
)

func unzip(f string) {
	r, _ := zip.OpenReader(f)
	for _, f := range r.File {
		p, _ := filepath.Abs(f.Name)
		// BAD: This could overwrite any file on the file system
		ioutil.WriteFile(p, []byte("present"), 0666)
	}
}

```
To fix this vulnerability, we need to check that the path does not contain any "`..`" elements in it.


```go
package main

import (
	"archive/zip"
	"io/ioutil"
	"path/filepath"
	"strings"
)

func unzipGood(f string) {
	r, _ := zip.OpenReader(f)
	for _, f := range r.File {
		p, _ := filepath.Abs(f.Name)
		// GOOD: Check that path does not contain ".." before using it
		if !strings.Contains(f.Name, "..") {
			ioutil.WriteFile(p, []byte("present"), 0666)
		}
	}
}

```

## References
* Snyk: [Zip Slip Vulnerability](https://snyk.io/research/zip-slip-vulnerability).
* OWASP: [Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* Common Weakness Enumeration: [CWE-22](https://cwe.mitre.org/data/definitions/22.html).

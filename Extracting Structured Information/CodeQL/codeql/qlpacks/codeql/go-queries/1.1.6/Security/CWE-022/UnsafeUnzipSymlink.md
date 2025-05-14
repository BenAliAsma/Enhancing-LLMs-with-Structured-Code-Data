# Arbitrary file write extracting an archive containing symbolic links
Extracting symbolic links from a malicious zip archive, without validating that the destination file path is within the destination directory, can cause files outside the destination directory to be overwritten. This can happen if there are previously-extracted symbolic links or directory traversal elements and links (`..`) in archive paths.

This problem is related to the ZipSlip vulnerability which is detected by the `go/zipslip` query; please see that query's help for more general information about malicious archive file vulnerabilities. This query considers the specific case where symbolic links are extracted from an archive, in which case the extraction code must be aware of existing symbolic links when checking whether it is about to extract a link pointing to a location outside the target extraction directory.


## Recommendation
Ensure that output paths constructed from zip archive entries are validated. This includes resolving any previously extracted symbolic links, for example using `path/filepath.EvalSymlinks`, to prevent writing files or links to unexpected locations.


## Example
In this example, links are extracted from an archive using the syntactic `filepath.Rel` function to check whether the link and its target fall within the destination directory. However, the extraction code doesn't resolve previously-extracted links, so a pair of links like `subdir/parent -> ..` followed by `escape -> subdir/parent/.. -> subdir/../..` leaves a link pointing to the parent of the archive root. The syntactic `Rel` is ineffective because it equates `subdir/parent/..` with `subdir/`, but this is not the case when `subdir/parent` is a symbolic link.


```go
package main

import (
	"archive/tar"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func isRel(candidate, target string) bool {
	// BAD: purely syntactic means are used to check
	// that `candidate` does not escape from `target`
	if filepath.IsAbs(candidate) {
		return false
	}
	relpath, err := filepath.Rel(target, filepath.Join(target, candidate))
	return err == nil && !strings.HasPrefix(filepath.Clean(relpath), "..")
}

func unzipSymlink(f io.Reader, target string) {
	r := tar.NewReader(f)
	for {
		header, err := r.Next()
		if err != nil {
			break
		}
		if isRel(header.Linkname, target) && isRel(header.Name, target) {
			os.Symlink(header.Linkname, header.Name)
		}
	}
}

```
To fix this vulnerability, resolve pre-existing symbolic links before checking that the link's target is acceptable:


```go
package main

func isRel(candidate, target string) bool {
	// GOOD: resolves all symbolic links before checking
	// that `candidate` does not escape from `target`
	if filepath.IsAbs(candidate) {
		return false
	}
	realpath, err := filepath.EvalSymlinks(filepath.Join(target, candidate))
	if err != nil {
		return false
	}
	relpath, err := filepath.Rel(target, realpath)
	return err == nil && !strings.HasPrefix(filepath.Clean(relpath), "..")
}

```

## References
* Snyk: [Zip Slip Vulnerability](https://snyk.io/research/zip-slip-vulnerability).
* OWASP: [Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* Common Weakness Enumeration: [CWE-22](https://cwe.mitre.org/data/definitions/22.html).

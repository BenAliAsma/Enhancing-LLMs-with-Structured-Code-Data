# Uncontrolled file decompression
Extracting Compressed files with any compression algorithm like gzip can cause to denial of service attacks.

Attackers can compress a huge file which created by repeated similiar byte and convert it to a small compressed file.


## Recommendation
When you want to decompress a user-provided compressed file you must be careful about the decompression ratio or read these files within a loop byte by byte to be able to manage the decompressed size in each cycle of the loop. Also you can limit the size of reader buffer.


## Example
Using "io.LimitReader" and "io.CopyN" are the best option to prevent decompression bomb attacks.


```go
package main

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
)

func ZipOpenReader(filename string) {
	// Open the zip file
	r, _ := zip.OpenReader(filename)
	var totalBytes int64
	for _, f := range r.File {
		rc, _ := f.Open()
		totalBytes = 0
		for {
			result, _ := io.CopyN(os.Stdout, rc, 68)
			if result == 0 {
				break
			}
			totalBytes = totalBytes + result
			if totalBytes > 1024*1024 {
				fmt.Print(totalBytes)
				_ = rc.Close()
				break
			}
		}
	}
}

```

```go
package main

import (
	"compress/gzip"
	"io"
	"os"
)

func safeReader() {
	var src io.Reader
	src, _ = os.Open("filename")
	gzipR, _ := gzip.NewReader(src)
	dstF, _ := os.OpenFile("./test", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0755)
	defer dstF.Close()
	var newSrc io.Reader
	newSrc = io.LimitReader(gzipR, 1024*1024*1024*5)
	_, _ = io.Copy(dstF, newSrc)
}

```

## References
* [CVE-2023-26483 ](https://github.com/russellhaering/gosaml2/security/advisories/GHSA-6gc3-crp7-25w5)
* [A great research to gain more impact by this kind of attacks](https://www.bamsoftware.com/hacks/zipbomb/)
* Common Weakness Enumeration: [CWE-409](https://cwe.mitre.org/data/definitions/409.html).

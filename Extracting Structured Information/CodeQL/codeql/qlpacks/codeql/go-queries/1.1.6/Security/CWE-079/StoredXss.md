# Stored cross-site scripting
Directly using externally-controlled stored values (for example, file names or database contents) to create HTML content without properly sanitizing the input first, allows for a cross-site scripting vulnerability.

This kind of vulnerability is also called *stored* cross-site scripting, to distinguish it from other types of cross-site scripting.


## Recommendation
To guard against cross-site scripting, consider using contextual output encoding/escaping before using uncontrolled stored values to create HTML content, or one of the other solutions that are mentioned in the references.


## Example
The following example code writes file names directly to an HTTP response. This leaves the website vulnerable to cross-site scripting, if an attacker can choose the file names on the disk.


```go
package main

import (
	"io"
	"io/ioutil"
	"net/http"
)

func ListFiles(w http.ResponseWriter, r *http.Request) {
	files, _ := ioutil.ReadDir(".")

	for _, file := range files {
		io.WriteString(w, file.Name()+"\n")
	}
}

```
Sanitizing the file names prevents the vulnerability:


```go
package main

import (
	"html"
	"io"
	"io/ioutil"
	"net/http"
)

func ListFiles1(w http.ResponseWriter, r *http.Request) {
	files, _ := ioutil.ReadDir(".")

	for _, file := range files {
		io.WriteString(w, html.EscapeString(file.Name())+"\n")
	}
}

```

## References
* OWASP: [XSS (Cross Site Scripting) Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html).
* OWASP: [Types of Cross-Site Scripting](https://www.owasp.org/index.php/Types_of_Cross-Site_Scripting).
* Wikipedia: [Cross-site scripting](http://en.wikipedia.org/wiki/Cross-site_scripting).
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).

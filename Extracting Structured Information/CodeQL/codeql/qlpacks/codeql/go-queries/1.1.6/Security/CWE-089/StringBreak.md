# Potentially unsafe quoting
Code that constructs a quoted string literal containing user-provided data needs to ensure that this data does not itself contain a quote. Otherwise the embedded data could (accidentally or intentionally) terminate the string literal early and thereby change the structure of the overall string, with potentially severe consequences. If, for example, the string is later used as part of an operating-system command or database query, an attacker may be able to craft input data that injects a malicious command.


## Recommendation
Sanitize the embedded data appropriately to ensure quotes are escaped, or use an API that does not rely on manually constructing quoted substrings. Make sure to use the appropriate escaping mechanism, for example, double quoting for SQL strings or backslash escaping for shell commands. When using backslash escaping, the backslash character itself must also be escaped.


## Example
In the following example, assume that `version` is an object from an untrusted source. The code snippet first uses `json.Marshal` to serialize this object into a string, and then embeds it into a SQL query built using the Squirrel library.


```go
package main

import (
	"encoding/json"
	"fmt"
	sq "github.com/Masterminds/squirrel"
)

func save(id string, version interface{}) {
	versionJSON, _ := json.Marshal(version)
	sq.StatementBuilder.
		Insert("resources").
		Columns("resource_id", "version_md5").
		Values(id, sq.Expr(fmt.Sprintf("md5('%s')", versionJSON))).
		Exec()
}

```
Note that JSON encoding does not escape single quotes in any way, so this code is vulnerable: any single-quote character in `version` will prematurely close the surrounding string literal, changing the structure of the SQL expression being constructed. This could be exploited to mount a SQL injection attack.

To fix this vulnerability, use the placeholder syntax from Squirrel's structured API for building queries, which avoids the need to explicitly construct a quoted string.


```go
package main

import (
	"encoding/json"
	sq "github.com/Masterminds/squirrel"
)

func saveGood(id string, version interface{}) {
	versionJSON, _ := json.Marshal(version)
	sq.StatementBuilder.
		Insert("resources").
		Columns("resource_id", "version_md5").
		Values(id, sq.Expr("md5(?)", versionJSON)).
		Exec()
}

```
In situations where a structured API is not available, make sure that you escape quotes before embedding user-provided data into a quoted string. For example, this is how you can backslash-escape single quotes using `strings.ReplaceAll`:

```go

  quoted := strings.ReplaceAll(raw, `\`, `\\`)
  quoted = strings.ReplaceAll(quoted, "'", "\\'")

```
Note that any existing backslash characters in the string must be escaped first, so that they do not interfere with the escaping of single quotes.

In some cases, `strconv.Quote` is a convenient option for backslash escaping, but note that it has two limitations:

1. It only supports double quotes, not single quotes (as in the example).
1. It puts quotes around the entire string, so it can only be used to construct complete string literals, not parts of larger string literals.

## References
* Wikipedia: [SQL injection](https://en.wikipedia.org/wiki/SQL_injection).
* OWASP: [Command Injection](https://www.owasp.org/index.php/Command_Injection).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).
* Common Weakness Enumeration: [CWE-89](https://cwe.mitre.org/data/definitions/89.html).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).

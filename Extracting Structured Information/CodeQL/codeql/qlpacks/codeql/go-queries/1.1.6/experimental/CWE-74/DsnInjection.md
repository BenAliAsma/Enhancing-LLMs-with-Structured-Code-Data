# SQL Data-source URI built from user-controlled sources
If a Data-Source Name (DSN) is built using untrusted user input without proper sanitization, the system may be vulnerable to DSN injection vulnerabilities.


## Recommendation
If user input must be included in a DSN, additional steps should be taken to sanitize untrusted data, such as checking for special characters included in user input.


## Example
In the following examples, the code accepts the db name from the user, which it then uses to build a DSN string.

The following example uses the unsanitized user input directly in the process of constructing a DSN name. A malicious user could provide special characters to change the meaning of this string, and carry out unexpected database operations.


```go
package main

import (
	"database/sql"
	"fmt"
	"os"
)

func bad() interface{} {
	name := os.Args[1:]
	// This is bad. `name` can be something like `test?allowAllFiles=true&` which will allow an attacker to access local files.
	dbDSN := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=utf8", "username", "password", "127.0.0.1", 3306, name)
	db, _ := sql.Open("mysql", dbDSN)
	return db
}

```
In the following example, the input provided by the user is sanitized before it is included in the DSN string. This ensures the meaning of the DSN string cannot be changed by a malicious user.


```go
package main

import (
	"database/sql"
	"errors"
	"fmt"
	"os"
	"regexp"
)

func good() (interface{}, error) {
	name := os.Args[1]
	hasBadChar, _ := regexp.MatchString(".*[?].*", name)

	if hasBadChar {
		return nil, errors.New("Bad input")
	}

	dbDSN := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=utf8", "username", "password", "127.0.0.1", 3306, name)
	db, _ := sql.Open("mysql", dbDSN)
	return db, nil
}

```

## References
* CVE-2022-3023: [Data Source Name Injection in pingcap/tidb.](https://nvd.nist.gov/vuln/detail/CVE-2022-3023/)
* Common Weakness Enumeration: [CWE-74](https://cwe.mitre.org/data/definitions/74.html).

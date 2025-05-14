# Improper LDAP Authentication
If an LDAP connection uses user-supplied data as password, anonymous bind could be caused using an empty password to result in a successful authentication.


## Recommendation
Don't use user-supplied data as password while establishing an LDAP connection.


## Example
In the following examples, the code accepts a bind password via a HTTP request in variable ` bindPassword`. The code builds a LDAP query whose authentication depends on user supplied data.


```go
package main

import (
	"fmt"
	"log"
)

func bad() interface{} {
	bindPassword := req.URL.Query()["password"][0]

	// Connect to the LDAP server
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatalf("Failed to connect to LDAP server: %v", err)
	}
	defer l.Close()

	err = l.Bind("cn=admin,dc=example,dc=com", bindPassword)
	if err != nil {
		log.Fatalf("LDAP bind failed: %v", err)
	}
}

```
In the following examples, the code accepts a bind password via a HTTP request in variable ` bindPassword`. The function ensures that the password provided is not empty before binding.


```go
package main

import (
	"fmt"
	"log"
)

func good() interface{} {
	bindPassword := req.URL.Query()["password"][0]

	// Connect to the LDAP server
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatalf("Failed to connect to LDAP server: %v", err)
	}
	defer l.Close()

	if bindPassword != "" {
		err = l.Bind("cn=admin,dc=example,dc=com", bindPassword)
		if err != nil {
			log.Fatalf("LDAP bind failed: %v", err)
		}
	}
}

```

## References
* MITRE: [CWE-287: Improper Authentication](https://cwe.mitre.org/data/definitions/287.html).
* Common Weakness Enumeration: [CWE-287](https://cwe.mitre.org/data/definitions/287.html).

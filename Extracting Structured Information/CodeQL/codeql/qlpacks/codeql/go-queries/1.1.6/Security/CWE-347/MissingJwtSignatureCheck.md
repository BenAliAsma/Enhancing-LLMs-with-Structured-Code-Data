# Missing JWT signature check
Applications decoding a JSON Web Token (JWT) may be vulnerable when the signature is not correctly verified.


## Recommendation
Always verify the signature by using the appropriate methods provided by the JWT library, or use a library that verifies it by default.


## Example
The following (bad) example shows a case where a JWT is parsed without verifying the signature.


```go
package main

import (
	"fmt"
	"log"

	"github.com/golang-jwt/jwt/v5"
)

type User struct{}

func decodeJwt(token string) {
	// BAD: JWT is only decoded without signature verification
	fmt.Println("only decoding JWT")
	DecodedToken, _, err := jwt.NewParser().ParseUnverified(token, &User{})
	if claims, ok := DecodedToken.Claims.(*User); ok {
		fmt.Printf("DecodedToken:%v\n", claims)
	} else {
		log.Fatal("error", err)
	}
}

```
The following (good) example uses the appropriate function for parsing a JWT and verifying its signature.


```go
package main

import (
	"fmt"
	"log"

	"github.com/golang-jwt/jwt/v5"
)

type User struct{}

func parseJwt(token string, jwtKey []byte) {
	// GOOD: JWT is parsed with signature verification using jwtKey
	DecodedToken, err := jwt.ParseWithClaims(token, &User{}, func(token *jwt.Token) (interface{}, error) {
		return jwtKey, nil
	})
	if claims, ok := DecodedToken.Claims.(*User); ok && DecodedToken.Valid && !err {
		fmt.Printf("DecodedToken:%v\n", claims)
	} else {
		log.Fatal(err)
	}
}

```

## References
* JWT IO: [Introduction to JSON Web Tokens](https://jwt.io/introduction).
* jwt-go: [Documentation](https://pkg.go.dev/github.com/golang-jwt/jwt/v5).
* Go JOSE: [Documentation](https://pkg.go.dev/github.com/go-jose/go-jose/v3).
* Common Weakness Enumeration: [CWE-347](https://cwe.mitre.org/data/definitions/347.html).

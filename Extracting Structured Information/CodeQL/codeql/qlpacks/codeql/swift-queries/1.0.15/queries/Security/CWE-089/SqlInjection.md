# Database query built from user-controlled sources
If a database query (such as a SQL query) is built from user-provided data without sufficient sanitization, a user may be able to run malicious database queries. An attacker can craft the part of the query they control to change the overall meaning of the query.


## Recommendation
Most database connector libraries offer a way to safely embed untrusted data into a query using query parameters or prepared statements. You should use these features to build queries, rather than string concatenation or similar methods. You can also escape (sanitize) user-controlled strings so that they can be included directly in an SQL command. A library function should be used for escaping, because this approach is only safe if the escaping function is robust against all possible inputs.


## Example
In the following examples, an SQL query is prepared using string interpolation to directly include a user-controlled value `userControlledString` in the query. An attacker could craft `userControlledString` to change the overall meaning of the SQL query.


```swift
// with SQLite.swift

let unsafeQuery = "SELECT * FROM users WHERE username='\(userControlledString)'"

try db.execute(unsafeQuery) // BAD

let stmt = try db.prepare(unsafeQuery) // also BAD
try stmt.run()

// with SQLite3 C API

let result = sqlite3_exec(db, unsafeQuery, nil, nil, nil) // BAD

```
A better way to do this is with a prepared statement, binding `userControlledString` to a parameter of that statement. An attacker who controls `userControlledString` now cannot change the overall meaning of the query.


```swift
// with SQLite.swift

let safeQuery = "SELECT * FROM users WHERE username=?"

let stmt = try db.prepare(safeQuery, userControlledString) // GOOD
try stmt.run()

// with sqlite3 C API

var stmt2: OpaquePointer?

if (sqlite3_prepare_v2(db, safeQuery, -1, &stmt2, nil) == SQLITE_OK) {
	if (sqlite3_bind_text(stmt2, 1, userControlledString, -1, SQLITE_TRANSIENT) == SQLITE_OK) { // GOOD
		let result = sqlite3_step(stmt2)

		// ...
	}
	sqlite3_finalize(stmt2)
}

```

## References
* Wikipedia: [SQL injection](https://en.wikipedia.org/wiki/SQL_injection).
* OWASP: [SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-89](https://cwe.mitre.org/data/definitions/89.html).

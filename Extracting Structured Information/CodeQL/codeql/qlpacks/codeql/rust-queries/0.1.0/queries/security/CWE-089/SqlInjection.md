# Database query built from user-controlled sources
If a database query (such as an SQL query) is built from user-provided data without sufficient sanitization, a user may be able to run malicious database queries. An attacker can craft the part of the query they control to change the overall meaning of the query.


## Recommendation
Most database connector libraries offer a way to safely embed untrusted data into a query using query parameters or prepared statements. You should use these features to build queries, rather than string concatenation or similar methods. You can also escape (sanitize) user-controlled strings so that they can be included directly in an SQL command. A library function should be used for escaping, because this approach is only safe if the escaping function is robust against all possible inputs.


## Example
In the following examples, an SQL query is prepared using string formatting to directly include a user-controlled value `remote_controlled_string`. An attacker could craft `remote_controlled_string` to change the overall meaning of the SQL query.


```rust
// with SQLx

let unsafe_query = format!("SELECT * FROM people WHERE firstname='{remote_controlled_string}'");

let _ = conn.execute(unsafe_query.as_str()).await?; // BAD (arbitrary SQL injection is possible)

let _ = sqlx::query(unsafe_query.as_str()).fetch_all(&mut conn).await?; // BAD (arbitrary SQL injection is possible)

```
A better way to do this is with a prepared statement, binding `remote_controlled_string` to a parameter of that statement. An attacker who controls `remote_controlled_string` now cannot change the overall meaning of the query.


```rust
// with SQLx

let prepared_query = "SELECT * FROM people WHERE firstname=?";

let _ = sqlx::query(prepared_query_1).bind(&remote_controlled_string).fetch_all(&mut conn).await?; // GOOD (prepared statement with bound parameter)

```

## References
* Wikipedia: [SQL injection](https://en.wikipedia.org/wiki/SQL_injection).
* OWASP: [SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-89](https://cwe.mitre.org/data/definitions/89.html).

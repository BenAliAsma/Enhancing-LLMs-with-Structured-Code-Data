# Cleartext storage of sensitive information in an SQLite database
Sensitive information that is stored in an unencrypted SQLite database is accessible to an attacker who gains access to the database.


## Recommendation
Ensure that if sensitive information is stored in a database then the database is always encrypted.


## Example
The following example shows two ways of storing information in an SQLite database. In the 'BAD' case, the credentials are simply stored in cleartext. In the 'GOOD' case, the database (and thus the credentials) are encrypted.


```c

void bad(void) {
  const char *password = "cleartext password";
  sqlite3 *credentialsDB;
  sqlite3_stmt *stmt;

  if (sqlite3_open("credentials.db", &credentialsDB) == SQLITE_OK) {
    // BAD: database opened without encryption being enabled
    sqlite3_exec(credentialsDB, "CREATE TABLE IF NOT EXISTS creds (password TEXT);", NULL, NULL, NULL);
    if (sqlite3_prepare_v2(credentialsDB, "INSERT INTO creds(password) VALUES(?)", -1, &stmt, NULL) == SQLITE_OK) {
      sqlite3_bind_text(stmt, 1, password, -1, SQLITE_TRANSIENT);
      sqlite3_step(stmt);
      sqlite3_finalize(stmt);
      sqlite3_close(credentialsDB);
    }
  }
}

void good(const char *secretKey) {
  const char *password = "cleartext password";
  sqlite3 *credentialsDB;
  sqlite3_stmt *stmt;

  if (sqlite3_open("credentials.db", &credentialsDB) == SQLITE_OK) {
    // GOOD: database encryption enabled:
    std::string setKeyString = std::string("PRAGMA key = '") + secretKey + "'";
    sqlite3_exec(credentialsDB, setKeyString.c_str(), NULL, NULL, NULL);
    sqlite3_exec(credentialsDB, "CREATE TABLE IF NOT EXISTS creds (password TEXT);", NULL, NULL, NULL);
    if (sqlite3_prepare_v2(credentialsDB, "INSERT INTO creds(password) VALUES(?)", -1, &stmt, NULL) == SQLITE_OK) {
      sqlite3_bind_text(stmt, 1, password, -1, SQLITE_TRANSIENT);
      sqlite3_step(stmt);
      sqlite3_finalize(stmt);
      sqlite3_close(credentialsDB);
    }
  }
}

```
Note that for the 'GOOD' example to work we need to provide a secret key. Secure key generation and storage is required.


## References
* M. Dowd, J. McDonald and J. Schuhm, *The Art of Software Security Assessment*, 1st Edition, Chapter 2 - 'Common Vulnerabilities of Encryption', p. 43. Addison Wesley, 2006.
* M. Howard and D. LeBlanc, *Writing Secure Code*, 2nd Edition, Chapter 9 - 'Protecting Secret Data', p. 299. Microsoft, 2002.
* Common Weakness Enumeration: [CWE-313](https://cwe.mitre.org/data/definitions/313.html).

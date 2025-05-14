# Clear-text logging of sensitive information
Sensitive information that is stored unencrypted is accessible to an attacker who gains access to the storage.


## Recommendation
Ensure that sensitive information is always encrypted before being stored.

In general, decrypt sensitive information only at the point where it is necessary for it to be used in cleartext.

Be aware that external processes often store the `standard out` and `standard error` streams of the application, causing logged sensitive information to be stored as well.


## Example
The following example code logs user credentials (in this case, their password) to `standard out` in plaintext:


```ruby
require 'Logger'

class UserSession
  @@logger = Logger.new STDOUT

  def login(username, password)
    # ...
    @@logger.info "login with password: #{password})"
  end
end

```
Instead, the credentials should be masked or redacted before logging:


```ruby
require 'Logger'

class UserSession
  @@logger = Logger.new STDOUT

  def login(username, password)
    # ...
    password_escaped = password.sub(/.*/, "[redacted]")
    @@logger.info "login with password: #{password_escaped})"
  end
end

```

## References
* M. Dowd, J. McDonald and J. Schuhm, *The Art of Software Security Assessment*, 1st Edition, Chapter 2 - 'Common Vulnerabilities of Encryption', p. 43. Addison Wesley, 2006.
* M. Howard and D. LeBlanc, *Writing Secure Code*, 2nd Edition, Chapter 9 - 'Protecting Secret Data', p. 299. Microsoft, 2002.
* Common Weakness Enumeration: [CWE-312](https://cwe.mitre.org/data/definitions/312.html).
* Common Weakness Enumeration: [CWE-359](https://cwe.mitre.org/data/definitions/359.html).
* Common Weakness Enumeration: [CWE-532](https://cwe.mitre.org/data/definitions/532.html).

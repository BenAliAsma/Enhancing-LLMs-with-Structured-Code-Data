# LDAP Injection
If an LDAP query or DN is built using string concatenation or string formatting, and the components of the concatenation include user input without any proper sanitization, a user is likely to be able to run malicious LDAP queries.


## Recommendation
If user input must be included in an LDAP query or DN, it should be escaped to avoid a malicious user providing special characters that change the meaning of the query.


## Example
In the following Rails example, an `ActionController` class has a `ldap_handler` method to handle requests.

The user and dc is specified using a parameter, `user_name` and `dc` provided by the client which it then uses to build a LDAP query and DN. This value is accessible using the `params` method.

The first example uses the unsanitized user input directly in the search filter and DN for the LDAP query. A malicious user could provide special characters to change the meaning of these components, and search for a completely different set of values.


```ruby
require 'net/ldap'

class BadLdapController < ActionController::Base
  def ldap_handler
    name = params[:user_name]
    dc = params[:dc]
    ldap = Net::LDAP.new(
        host: 'ldap.example.com',
        port: 636,
        encryption: :simple_tls,
        auth: {
            method: :simple,
            username: 'uid=admin,dc=example,dc=com',
            password: 'adminpassword'
        }
    )
    filter = Net::LDAP::Filter.eq('foo', name)
    attrs = [name]
    result = ldap.search(base: "ou=people,dc=#{dc},dc=com", filter: filter, attributes: attrs)
  end
end

```
In the second example, the input provided by the user is sanitized before it is included in the search filter or DN. This ensures the meaning of the query cannot be changed by a malicious user.


```ruby
require 'net/ldap'

class GoodLdapController < ActionController::Base
  def ldap_handler
    name = params[:user_name]
    ldap = Net::LDAP.new(
        host: 'ldap.example.com',
        port: 636,
        encryption: :simple_tls,
        auth: {
            method: :simple,
            username: 'uid=admin,dc=example,dc=com',
            password: 'adminpassword'
        }
    )
    
    name = if ["admin", "guest"].include? name
      name
    else 
      name = "none"
    end

    filter = Net::LDAP::Filter.eq('foo', name)
    attrs = ['dn']
    result = ldap.search(base: 'ou=people,dc=example,dc=com', filter: filter, attributes: attrs)
  end
end

```

## References
* OWASP: [LDAP Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LDAP_Injection_Prevention_Cheat_Sheet.html).
* OWASP: [LDAP Injection](https://owasp.org/www-community/attacks/LDAP_Injection).
* Wikipedia: [LDAP injection](https://en.wikipedia.org/wiki/LDAP_injection).
* BlackHat: [LDAP Injection and Blind LDAP Injection](https://www.blackhat.com/presentations/bh-europe-08/Alonso-Parada/Whitepaper/bh-eu-08-alonso-parada-WP.pdf).
* LDAP: [Understanding and Defending Against LDAP Injection Attacks](https://ldap.com/2018/05/04/understanding-and-defending-against-ldap-injection-attacks/).
* Common Weakness Enumeration: [CWE-90](https://cwe.mitre.org/data/definitions/90.html).

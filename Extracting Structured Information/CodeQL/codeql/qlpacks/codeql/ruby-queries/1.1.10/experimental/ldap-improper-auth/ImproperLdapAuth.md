# Improper LDAP Authentication
If an LDAP connection uses user-supplied data as password, anonymous bind could be caused using an empty password to result in a successful authentication.


## Recommendation
Don't use user-supplied data as password while establishing an LDAP connection.


## Example
In the following Rails example, an `ActionController` class has a `ldap_handler` method to handle requests.

In the first example, the code builds a LDAP query whose authentication depends on user supplied data.


```ruby
class FooController < ActionController::Base
  def some_request_handler
    pass = params[:pass]
    ldap = Net::LDAP.new(
        host: 'ldap.example.com',
        port: 636,
        encryption: :simple_tls,
        auth: {
            method: :simple,
            username: 'uid=admin,dc=example,dc=com',
            password: pass
        }
    )
    ldap.bind
  end
end
```
In the second example, the authentication is established using a default password.


```ruby
class FooController < ActionController::Base
  def some_request_handler
    pass = params[:pass]
    ldap = Net::LDAP.new(
        host: 'ldap.example.com',
        port: 636,
        encryption: :simple_tls,
        auth: {
            method: :simple,
            username: 'uid=admin,dc=example,dc=com',
            password: '$uper$password123'
        }
    )
    ldap.bind
  end
end
```

## References
* MITRE: [CWE-287: Improper Authentication](https://cwe.mitre.org/data/definitions/287.html).
* Common Weakness Enumeration: [CWE-287](https://cwe.mitre.org/data/definitions/287.html).

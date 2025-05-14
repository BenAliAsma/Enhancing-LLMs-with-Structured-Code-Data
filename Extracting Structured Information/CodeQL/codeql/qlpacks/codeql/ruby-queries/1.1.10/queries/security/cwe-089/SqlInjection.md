# SQL query built from user-controlled sources
If a database query (such as a SQL or NoSQL query) is built from user-provided data without sufficient sanitization, a malicious user may be able to run malicious database queries.


## Recommendation
Most database connector libraries offer a way of safely embedding untrusted data into a query by means of query parameters or prepared statements.


## Example
In the following Rails example, an `ActionController` class has a `text_bio` method to handle requests to fetch a biography for a specified user.

The user is specified using a parameter, `user_name` provided by the client. This value is accessible using the `params` method.

The method illustrates three different ways to construct and execute an SQL query to find the user by name.

In the first case, the parameter `user_name` is inserted into an SQL fragment using string interpolation. The parameter is user-supplied and is not sanitized. An attacker could use this to construct SQL queries that were not intended to be executed here.

The second case uses string concatenation and is vulnerable in the same way that the first case is.

In the third case, the name is passed in a hash instead. `ActiveRecord` will construct a parameterized SQL query that is not vulnerable to SQL injection attacks.


```ruby
class UserController < ActionController::Base
  def text_bio
    # BAD -- Using string interpolation
    user = User.find_by "name = '#{params[:user_name]}'"

    # BAD -- Using string concatenation
    find_str = "name = '" + params[:user_name] + "'"
    user = User.find_by(find_str)

    # GOOD -- Using a hash to parameterize arguments
    user = User.find_by name: params[:user_name]

    render plain: user&.text_bio
  end
end

```

## References
* Wikipedia: [SQL injection](https://en.wikipedia.org/wiki/SQL_injection).
* OWASP: [SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-89](https://cwe.mitre.org/data/definitions/89.html).

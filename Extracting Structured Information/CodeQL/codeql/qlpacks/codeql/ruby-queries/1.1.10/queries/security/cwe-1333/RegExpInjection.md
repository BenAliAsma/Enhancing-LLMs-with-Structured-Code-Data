# Regular expression injection
Constructing a regular expression with unsanitized user input is dangerous, since a malicious user may be able to modify the meaning of the expression. In particular, such a user may be able to provide a regular expression fragment that takes exponential time in the worst case, and use that to perform a Denial of Service attack.


## Recommendation
Before embedding user input into a regular expression, use a sanitization function such as `Regexp.escape` to escape meta-characters that have special meaning.


## Example
The following examples construct regular expressions from an HTTP request parameter without sanitizing it first:


```ruby
class UsersController < ActionController::Base
  def first_example
    # BAD: Unsanitized user input is used to construct a regular expression
    regex = /#{ params[:key] }/
  end

  def second_example
    # BAD: Unsanitized user input is used to construct a regular expression
    regex = Regexp.new(params[:key])
  end
end
```
Instead, the request parameter should be sanitized first. This ensures that the user cannot insert characters that have special meanings in regular expressions.


```ruby
class UsersController < ActionController::Base
  def example
    # GOOD: User input is sanitized before constructing the regular expression
    regex = Regexp.new(Regex.escape(params[:key]))
  end
end
```

## References
* OWASP: [Regular expression Denial of Service - ReDoS](https://www.owasp.org/index.php/Regular_expression_Denial_of_Service_-_ReDoS).
* Wikipedia: [ReDoS](https://en.wikipedia.org/wiki/ReDoS).
* Ruby: [Regexp.escape](https://ruby-doc.org/core-3.0.2/Regexp.html#method-c-escape).
* Common Weakness Enumeration: [CWE-1333](https://cwe.mitre.org/data/definitions/1333.html).
* Common Weakness Enumeration: [CWE-730](https://cwe.mitre.org/data/definitions/730.html).
* Common Weakness Enumeration: [CWE-400](https://cwe.mitre.org/data/definitions/400.html).

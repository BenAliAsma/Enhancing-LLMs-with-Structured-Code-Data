# Unsafe HTML constructed from library input
When a library function dynamically constructs HTML in a potentially unsafe way, then it's important to document to clients of the library that the function should only be used with trusted inputs. If the function is not documented as being potentially unsafe, then a client may inadvertently use inputs containing unsafe HTML fragments, and thereby leave the client vulnerable to cross-site scripting attacks.


## Recommendation
Document all library functions that can lead to cross-site scripting attacks, and guard against unsafe inputs where dynamic HTML construction is not intended.


## Example
The following example has a library function that renders a boldface name by creating a string containing a `<b>` with the name embedded in it.


```ruby
class UsersController < ActionController::Base
  # BAD - create a user description, where the name is not escaped
  def create_user_description (name)
    "<b>#{name}</b>".html_safe
  end
end

```
This library function, however, does not escape unsafe HTML, and a client that calls the function with user-supplied input may be vulnerable to cross-site scripting attacks.

The library could either document that this function should not be used with unsafe inputs, or escape the input before embedding it in the HTML fragment.


```ruby
class UsersController < ActionController::Base
  # Good - create a user description, where the name is escaped
  def create_user_description (name)
    "<b>#{ERB::Util.html_escape(name)}</b>".html_safe
  end
end

```

## References
* OWASP: [DOM based XSS Prevention Cheat Sheet](https://www.owasp.org/index.php/DOM_based_XSS_Prevention_Cheat_Sheet).
* OWASP: [XSS (Cross Site Scripting) Prevention Cheat Sheet](https://www.owasp.org/index.php/XSS_%28Cross_Site_Scripting%29_Prevention_Cheat_Sheet).
* OWASP [DOM Based XSS](https://www.owasp.org/index.php/DOM_Based_XSS).
* OWASP [Types of Cross-Site Scripting](https://www.owasp.org/index.php/Types_of_Cross-Site_Scripting).
* Wikipedia: [Cross-site scripting](http://en.wikipedia.org/wiki/Cross-site_scripting).
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).

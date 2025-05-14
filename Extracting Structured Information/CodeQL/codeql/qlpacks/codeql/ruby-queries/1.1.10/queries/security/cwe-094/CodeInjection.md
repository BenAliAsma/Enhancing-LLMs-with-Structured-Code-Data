# Code injection
Directly evaluating user input (for example, an HTTP request parameter) as code without first sanitizing the input allows an attacker arbitrary code execution. This can occur when user input is passed to code that interprets it as an expression to be evaluated, using methods such as `Kernel.eval` or `Kernel.send`.


## Recommendation
Avoid including user input in any expression that may be dynamically evaluated. If user input must be included, use context-specific escaping before including it. It is important that the correct escaping is used for the type of evaluation that will occur.


## Example
The following example shows two functions setting a name from a request. The first function uses `eval` to execute the `set_name` method. This is dangerous as it can allow a malicious user to execute arbitrary code on the server. For example, the user could supply the value `"' + exec('rm -rf') + '"` to destroy the server's file system. The second function calls the `set_name` method directly and is thus safe.


```ruby
class UsersController < ActionController::Base
  # BAD - Allow user to define code to be run.
  def create_bad
    first_name = params[:first_name]
    eval("set_name(#{first_name})")
  end

  # GOOD - Call code directly
  def create_good
    first_name = params[:first_name]
    set_name(first_name)
  end

  def set_name(name)
    @name = name
  end
end

```

## References
* OWASP: [Code Injection](https://www.owasp.org/index.php/Code_Injection).
* Wikipedia: [Code Injection](https://en.wikipedia.org/wiki/Code_injection).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
* Common Weakness Enumeration: [CWE-95](https://cwe.mitre.org/data/definitions/95.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).

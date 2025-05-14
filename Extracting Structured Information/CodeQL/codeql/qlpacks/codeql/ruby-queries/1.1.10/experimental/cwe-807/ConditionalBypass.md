# User-controlled bypass of security check
Using user-controlled data in a permissions check may allow a user to gain unauthorized access to protected functionality or data.


## Recommendation
When checking whether a user is authorized for a particular activity, do not use data that is entirely controlled by that user in the permissions check. If necessary, always validate the input, ideally against a fixed list of expected values.

Similarly, do not decide which permission to check based on user data. In particular, avoid using computation to decide which permissions to check for. Use fixed permissions for particular actions, rather than generating the permission to check for.


## Example
In this example, the controller decided whether or not to authenticate the user based on the value of a request parameter.


```ruby
class UsersController < ActionController::Base
  def example
    user = User.find_by(login: params[:login])
    if params[:authenticate]
      # BAD: decision to take sensitive action based on user-controlled data
      log_in user
      redirect_to user
    end
  end
end
```

## References
* Common Weakness Enumeration: [CWE-807](https://cwe.mitre.org/data/definitions/807.html).
* Common Weakness Enumeration: [CWE-290](https://cwe.mitre.org/data/definitions/290.html).

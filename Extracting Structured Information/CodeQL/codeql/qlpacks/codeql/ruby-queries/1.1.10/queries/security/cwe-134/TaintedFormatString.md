# Use of externally-controlled format string
Methods like `Kernel.printf` accept a format string that is used to format the remaining arguments by providing inline format specifiers. If the format string contains unsanitized input from an untrusted source, then that string may contain unexpected format specifiers that cause garbled output or throw an exception.


## Recommendation
Either sanitize the input before including it in the format string, or use a `%s` specifier in the format string, and pass the untrusted data as corresponding argument.


## Example
The following program snippet logs information about an unauthorized access attempt. The log message includes the user name, and the user's IP address is passed as an additional argument to `Kernel.printf` to be appended to the message:


```ruby
class UsersController < ActionController::Base
  def index
    printf("Unauthorised access attempt by #{params[:user]}: %s", request.ip)
  end
end
```
However, if a malicious user provides a format specified such as `%s` as their user name, `Kernel.printf` will throw an exception as there are too few arguments to satisfy the format. This can result in denial of service or leaking of internal information to the attacker via a stack trace.

Instead, the user name should be included using the `%s` specifier:


```ruby
class UsersController < ActionController::Base
  def index
    printf("Unauthorised access attempt by %s: %s", params[:user], request.ip)
  end
end
```
Alternatively, string interpolation should be used exclusively:


```ruby
class UsersController < ActionController::Base
  def index
    puts "Unauthorised access attempt by #{params[:user]}: #{request.ip}"
  end
end
```

## References
* Ruby documentation for [format strings](https://docs.ruby-lang.org/en/3.1/Kernel.html#method-i-sprintf).
* Common Weakness Enumeration: [CWE-134](https://cwe.mitre.org/data/definitions/134.html).

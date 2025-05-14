# Server-side template injection
Template Injection occurs when user input is embedded in a template's code in an unsafe manner. An attacker can use native template syntax to inject a malicious payload into a template, which is then executed server-side. This permits the attacker to run arbitrary code in the server's context.


## Recommendation
To fix this, ensure that untrusted input is not used as part of a template's code. If the application requirements do not allow this, use a sandboxed environment where access to unsafe attributes and methods is prohibited.


## Example
Consider the example given below, an untrusted HTTP parameter `name` is used to generate a template string. This can lead to remote code execution.


```ruby
require 'erb'
require 'slim'

class BadERBController < ActionController::Base
  def some_request_handler
    name = params["name"]
    html_text = "
      <!DOCTYPE html><html><body>
      <h2>Hello %s </h2></body></html>
      " % name
    template = ERB.new(html_text).result(binding) 
    render inline: html_text
  end
end

class BadSlimController < ActionController::Base
  def some_request_handler
    name = params["name"]
    html_text = "
      <!DOCTYPE html><html><body>
      <h2>Hello %s </h2></body></html>
      " % name
    Slim::Template.new{ html_text }.render 
  end
end
```
Here we have fixed the problem by including ERB/Slim syntax in the string, then the user input will be rendered but not evaluated.


```ruby
require 'erb'
require 'slim'

class GoodController < ActionController::Base
  def some_request_handler
    name = params["name"]
    html_text = "
      <!DOCTYPE html><html><body>
      <h2>Hello <%= name %> </h2></body></html>
      "
    template = ERB.new(html_text).result(binding) 
    render inline: html_text
  end
end

class GoodController < ActionController::Base
  def some_request_handler
    name = params["name"]
    html_text = "
    <!DOCTYPE html>
      html
        body
          h2  == name;
    "
    Slim::Template.new{ html_text }.render(Object.new, name: name)
  end
end
```

## References
* Wikipedia: [Server Side Template Injection](https://en.wikipedia.org/wiki/Code_injection#Server_Side_Template_Injection).
* Portswigger : [Server Side Template Injection](https://portswigger.net/web-security/server-side-template-injection).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).

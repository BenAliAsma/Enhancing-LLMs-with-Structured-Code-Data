# URL redirection from remote source
Directly incorporating user input into a URL redirect request without validating the input can facilitate phishing attacks. In these attacks, unsuspecting users can be redirected to a malicious site that looks very similar to the real site they intend to visit, but which is controlled by the attacker.


## Recommendation
To guard against untrusted URL redirection, it is advisable to avoid putting user input directly into a redirect URL. Instead, maintain a list of authorized redirects on the server; then choose from that list based on the user input provided.

If this is not possible, then the user input should be validated in some other way, for example, by verifying that the target URL is on the same host as the current page.


## Example
The following example shows an HTTP request parameter being used directly in a URL redirect without validating the input, which facilitates phishing attacks:


```ruby
class HelloController < ActionController::Base
  def hello
    redirect_to params[:url]
  end
end

```
One way to remedy the problem is to validate the user input against a set of known fixed strings before doing the redirection:


```ruby
class HelloController < ActionController::Base
  VALID_REDIRECTS = [
    "http://cwe.mitre.org/data/definitions/601.html",
    "http://cwe.mitre.org/data/definitions/79.html"
  ].freeze

  def hello
    # GOOD: the request parameter is validated against a known list of URLs
    target_url = params[:url]
    if VALID_REDIRECTS.include?(target_url)
      redirect_to target_url
    else
      redirect_to "/error.html"
    end
  end
end
```
Alternatively, we can check that the target URL does not redirect to a different host by checking that the URL is either relative or on a known good host:


```ruby
require 'uri'

class HelloController < ActionController::Base
  KNOWN_HOST = "example.org"

  def hello
    begin
      target_url = URI.parse(params[:url])

      # Redirect if the URL is either relative or on a known good host
      if !target_url.host || target_url.host == KNOWN_HOST
        redirect_to target_url.to_s
      else
        redirect_to "/error.html" # Redirect to error page if the host is not known
      end
    rescue URI::InvalidURIError
      # Handle the exception, for example, by redirecting to a safe page
      redirect_to "/error.html"
    end
  end
end
```
Note that as written, the above code will allow redirects to URLs on `example.com`, which is harmless but perhaps not intended. You can substitute your own domain (if known) for `example.com` to prevent this.


## References
* OWASP: [ Unvalidated Redirects and Forwards Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html).
* Rails Guides: [Redirection and Files](https://guides.rubyonrails.org/security.html#redirection-and-files).
* Common Weakness Enumeration: [CWE-601](https://cwe.mitre.org/data/definitions/601.html).

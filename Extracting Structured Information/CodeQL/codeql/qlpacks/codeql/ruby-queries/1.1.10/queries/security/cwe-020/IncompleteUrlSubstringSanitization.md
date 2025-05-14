# Incomplete URL substring sanitization
Sanitizing untrusted URLs is an important technique for preventing attacks such as request forgeries and malicious redirections. Usually, this is done by checking that the host of a URL is in a set of allowed hosts.

However, treating the URL as a string and checking if one of the allowed hosts is a substring of the URL is very prone to errors. Malicious URLs can bypass such security checks by embedding one of the allowed hosts in an unexpected location.

Even if the substring check is not used in a security-critical context, the incomplete check may still cause undesirable behaviors when the check succeeds accidentally.


## Recommendation
Parse a URL before performing a check on its host value, and ensure that the check handles arbitrary subdomain sequences correctly.


## Example
The following example code checks that a URL redirection will reach the `example.com` domain, or one of its subdomains, and not some malicious site.


```ruby
class AppController < ApplicationController
    def index
        url = params[:url]
        # BAD: the host of `url` may be controlled by an attacker
        if url.include?("example.com")
            redirect_to url
        end
    end
end
```
The substring check is, however, easy to bypass. For example by embedding `example.com` in the path component: `http://evil-example.net/example.com`, or in the query string component: `http://evil-example.net/?x=example.com`. Address these shortcomings by checking the host of the parsed URL instead:


```ruby
class AppController < ApplicationController
    def index
        url = params[:url]
        host = URI(url).host
        # BAD: the host of `url` may be controlled by an attacker
        if host.include?("example.com")
            redirect_to url
        end
    end
end

```
This is still not a sufficient check as the following URLs bypass it: `http://evil-example.com` `http://example.com.evil-example.net`. Instead, use an explicit whitelist of allowed hosts to make the redirect secure:


```ruby
class AppController < ApplicationController
    def index
        url = params[:url]
        host = URI(url).host
        # GOOD: the host of `url` can not be controlled by an attacker
        allowedHosts = [
            'example.com',
            'beta.example.com',
            'www.example.com'
        ]
        if allowedHosts.include?(host)
            redirect_to url
        end
    end
end

```

## References
* OWASP: [SSRF](https://www.owasp.org/index.php/Server_Side_Request_Forgery)
* OWASP: [XSS Unvalidated Redirects and Forwards Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).

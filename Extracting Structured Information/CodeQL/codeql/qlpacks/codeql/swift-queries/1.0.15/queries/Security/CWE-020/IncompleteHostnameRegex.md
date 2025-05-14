# Incomplete regular expression for hostnames
Sanitizing untrusted URLs is an important technique for preventing attacks such as request forgeries and malicious redirections. Often, this is done by checking that the host of a URL is in a set of allowed hosts.

If a regular expression implements such a check, it is easy to accidentally make the check too permissive by not escaping the `.` meta-characters appropriately. Even if the check is not used in a security-critical context, the incomplete check may still cause undesirable behaviors when it accidentally succeeds.


## Recommendation
Escape all meta-characters appropriately when constructing regular expressions for security checks, and pay special attention to the `.` meta-character.


## Example
The following example code checks that a URL redirection will reach the `example.com` domain, or one of its subdomains.


```

func handleUrl(_ urlString: String) {
    // get the 'url=' parameter from the URL
    let components = URLComponents(string: urlString)
    let redirectParam = components?.queryItems?.first(where: { $0.name == "url" })

    // check we trust the host
    let regex = #/^(www|beta).example.com//#  // BAD
    if let match = redirectParam?.value?.firstMatch(of: regex) {
        // ... trust the URL ...
    }
}

```
The check is, however, easy to bypass because the unescaped `.` allows for any character before `example.com`, effectively allowing the redirect to go to an attacker-controlled domain such as `wwwXexample.com`.

Address this vulnerability by escaping `.` to `\.`:


```

func handleUrl(_ urlString: String) {
    // get the 'url=' parameter from the URL
    let components = URLComponents(string: urlString)
    let redirectParam = components?.queryItems?.first(where: { $0.name == "url" })

    // check we trust the host
    let regex = #/^(www|beta)\.example\.com//#  // GOOD
    if let match = redirectParam?.value?.firstMatch(of: regex) {
        // ... trust the URL ...
    }
}

```

## References
* OWASP: [Server Side Request Forgery](https://www.owasp.org/index.php/Server_Side_Request_Forgery).
* OWASP: [Unvalidated Redirects and Forwards Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).

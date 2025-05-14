# Incorrect URL verification
Apps that rely on URL Parsing to verify that a given URL is pointing to a trust server may be susceptible to many different ways to get URL parsing and verification wrong, which allows an attacker to register a fake site to break the access control.


## Recommendation
Verify the whole host and domain (FQDN) or check endsWith dot+domain.


## Example
The following example shows two ways of verifying host domain. In the 'BAD' case, verification is implemented as partial domain match. In the 'GOOD' case, full domain is verified.


```java
public boolean shouldOverrideUrlLoading(WebView view, String url) {
    {
        Uri uri = Uri.parse(url);
        // BAD: partial domain match, which allows an attacker to register a domain like myexample.com to circumvent the verification
        if (uri.getHost() != null && uri.getHost().endsWith("example.com")) {
            return false;
        }
    }

    {
        Uri uri = Uri.parse(url);
        // GOOD: full domain match
        if (uri.getHost() != null && uri.getHost().endsWith(".example.com")) {
            return false;
        }
    }
}

```

## References
* [Common Android app vulnerabilities from Sebastian Porst of Google](https://drive.google.com/file/d/0BwMN49Gzo3x6T1N5WGQ4TTNlMHBOb1ZRQTVEWnVBZjFUaE5N/view)
* [Common Android app vulnerabilities from bugcrowd](https://www.bugcrowd.com/resources/webinars/overview-of-common-android-app-vulnerabilities/)
* Common Weakness Enumeration: [CWE-939](https://cwe.mitre.org/data/definitions/939.html).

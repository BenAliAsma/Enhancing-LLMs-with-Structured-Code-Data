# Dependency download using unencrypted communication channel
Using an insecure protocol like HTTP or FTP to download dependencies makes the build process vulnerable to a man-in-the-middle (MITM) attack.

This can allow attackers to inject malicious code into the downloaded dependencies, and thereby infect the build artifacts and execute arbitrary code on the machine building the artifacts.


## Recommendation
Always use a secure protocol, such as HTTPS or SFTP, when downloading artifacts from a URL.


## Example
The below example shows a `Gemfile` that specifies a gem source using the insecure HTTP protocol.


```ruby
source "http://rubygems.org"

gem "my-gem-a", "1.2.3"
```
The fix is to change the protocol to HTTPS.


```ruby
source "https://rubygems.org"

gem "my-gem-a", "1.2.3"
```

## References
* Jonathan Leitschuh: [ Want to take over the Java ecosystem? All you need is a MITM! ](https://infosecwriteups.com/want-to-take-over-the-java-ecosystem-all-you-need-is-a-mitm-1fc329d898fb)
* Max Veytsman: [ How to take over the computer of any Java (or Clojure or Scala) Developer. ](https://max.computer/blog/how-to-take-over-the-computer-of-any-java-or-clojure-or-scala-developer/)
* Wikipedia: [Supply chain attack.](https://en.wikipedia.org/wiki/Supply_chain_attack)
* Wikipedia: [Man-in-the-middle attack.](https://en.wikipedia.org/wiki/Man-in-the-middle_attack)
* Common Weakness Enumeration: [CWE-300](https://cwe.mitre.org/data/definitions/300.html).
* Common Weakness Enumeration: [CWE-319](https://cwe.mitre.org/data/definitions/319.html).
* Common Weakness Enumeration: [CWE-494](https://cwe.mitre.org/data/definitions/494.html).
* Common Weakness Enumeration: [CWE-829](https://cwe.mitre.org/data/definitions/829.html).

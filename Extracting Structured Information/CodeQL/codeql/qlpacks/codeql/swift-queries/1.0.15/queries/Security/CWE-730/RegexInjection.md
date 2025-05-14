# Regular expression injection
Constructing a regular expression with unsanitized user input is dangerous, since a malicious user may be able to modify the meaning of the expression. They may be able to cause unexpected program behaviour, or perform a denial-of-service attack. For example, they may provide a regular expression fragment that takes exponential time to evaluate in the worst case.


## Recommendation
Before embedding user input into a regular expression, use a sanitization function such as `NSRegularExpression::escapedPattern(for:)` to escape meta-characters that have special meaning.


## Example
The following examples construct regular expressions from user input without sanitizing it first:


```swift
func processRemoteInput(remoteInput: String) {
  ...

  // BAD: Unsanitized user input is used to construct a regular expression
  let regex1 = try Regex(remoteInput)

  // BAD: Unsanitized user input is used to construct a regular expression
  let regexStr = "abc|\(remoteInput)"
  let regex2 = try NSRegularExpression(pattern: regexStr)

  ...
}

```
If user input is used to construct a regular expression it should be sanitized first. This ensures that the user cannot insert characters that have special meanings in regular expressions.


```swift
func processRemoteInput(remoteInput: String) {
  ...

  // GOOD: Regular expression is not derived from user input
  let regex1 = try Regex(myRegex)

  // GOOD: User input is sanitized before being used to construct a regular expression
  let escapedInput = NSRegularExpression.escapedPattern(for: remoteInput)
  let regexStr = "abc|\(escapedInput)"
  let regex2 = try NSRegularExpression(pattern: regexStr)

  ...
}

```

## References
* OWASP: [Regular expression Denial of Service - ReDoS](https://www.owasp.org/index.php/Regular_expression_Denial_of_Service_-_ReDoS).
* Wikipedia: [ReDoS](https://en.wikipedia.org/wiki/ReDoS).
* Swift: [NSRegularExpression.escapedPattern(for:)](https://developer.apple.com/documentation/foundation/nsregularexpression/1408386-escapedpattern).
* Common Weakness Enumeration: [CWE-730](https://cwe.mitre.org/data/definitions/730.html).
* Common Weakness Enumeration: [CWE-400](https://cwe.mitre.org/data/definitions/400.html).

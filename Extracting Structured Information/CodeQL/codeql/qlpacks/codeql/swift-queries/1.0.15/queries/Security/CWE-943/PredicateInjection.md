# Predicate built from user-controlled sources
Predicates represent logical conditions that can be used to check whether an object matches them. If a predicate is built from user-provided data without sufficient sanitization, an attacker may be able to change the overall meaning of the predicate.


## Recommendation
When building a predicate from untrusted data, you should either pass it to the appropriate `arguments` parameter during initialization, or as an array of substitution variables before evaluation. You should not append or concatenate it to the body of the predicate.


## Example
In the following insecure example, `NSPredicate` is built directly from data obtained from an HTTP request. This is untrusted, and can be arbitrarily set by an attacker to alter the meaning of the predicate:


```swift
let remoteString = try String(contentsOf: URL(string: "https://example.com/")!)

let filenames: [String] = ["img1.png", "img2.png", "img3.png", "img.txt", "img.csv"]

let predicate = NSPredicate(format: "SELF LIKE \(remoteString)")
let filtered = filenames.filter(){ filename in
    predicate.evaluate(with: filename)
}
print(filtered)

```
A better way to do this is to use the `arguments` parameter of `NSPredicate`'s initializer. This prevents attackers from altering the meaning of the predicate, even if they control the externally obtained data, as seen in the following secure example:


```swift
let remoteString = try String(contentsOf: URL(string: "https://example.com/")!)

let filenames: [String] = ["img1.png", "img2.png", "img3.png", "img.txt", "img.csv"]

let predicate = NSPredicate(format: "SELF LIKE %@", remoteString)
let filtered = filenames.filter(){ filename in
    predicate.evaluate(with: filename)
}
print(filtered)

```

## References
* Apple Developer Documentation: [NSPredicate](https://developer.apple.com/documentation/foundation/nspredicate)
* Common Weakness Enumeration: [CWE-943](https://cwe.mitre.org/data/definitions/943.html).

# String length conflation
Using a length value from an `NSString` in a `String`, or a count from a `String` in an `NSString`, may cause unexpected behavior including (in some cases) buffer overwrites. This is because certain unicode sequences are represented as one character in a `String` but as a sequence of multiple characters in an `NSString`. For example, a 'thumbs up' emoji with a skin tone modifier (👍🏿) is represented as U+1F44D (👍) then the modifier U+1F3FF.

This issue can also arise from using the values of `String.utf8.count`, `String.utf16.count` or `String.unicodeScalars.count` in an unsuitable place.


## Recommendation
Use `String.count` when working with a `String`. Use `NSString.length` when working with an `NSString`. Do not mix values for lengths and offsets between the two types as they are not compatible measures.

If you need to convert between `Range` and `NSRange`, do so directly using the appropriate initializer. Do not attempt to use incompatible length and offset values to accomplish conversion.


## Example
In the following example, a `String` is converted to `NSString`, but a range is created from the `String` to do some processing on it.


```swift

func myFunction(s: String) {
	let ns = NSString(string: s)
	let nsrange = NSMakeRange(0, s.count) // BAD: String length used in NSMakeRange

	// ... use nsrange to process ns
}

```
This is dangerous because, if the input contains certain characters, the range computed on the `String` will be wrong for the `NSString`. This will lead to incorrect behaviour in the string processing that follows. To fix the problem, we can use `NSString.length` to create the `NSRange` instead, as follows:


```swift

func myFunction(s: String) {
	let ns = NSString(string: s)
	let nsrange = NSMakeRange(0, ns.length) // Fixed: NSString length used in NSMakeRange

	// ... use nsrange to process ns
}

```

## References
* [Swift String vs. NSString](https://talk.objc.io/episodes/S01E80-swift-string-vs-nsstring)
* Common Weakness Enumeration: [CWE-135](https://cwe.mitre.org/data/definitions/135.html).

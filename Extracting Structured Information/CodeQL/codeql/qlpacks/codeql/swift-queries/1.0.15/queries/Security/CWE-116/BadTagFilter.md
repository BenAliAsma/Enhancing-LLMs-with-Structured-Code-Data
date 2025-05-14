# Bad HTML filtering regexp
It is possible to match some single HTML tags using regular expressions (parsing general HTML using regular expressions is impossible). However, if the regular expression is not written well, it might be possible to circumvent it. This can lead to cross-site scripting or other security issues.

Some of these mistakes are caused by browsers having very forgiving HTML parsers, and will often render invalid HTML containing syntax errors. Regular expressions that attempt to match HTML should also recognize tags containing such syntax errors.


## Recommendation
Use a well-tested sanitization or parser library if at all possible. These libraries are much more likely to handle corner cases correctly than a custom implementation.


## Example
The following example attempts to filters out all `<script>` tags.


```
let script_tag_regex = /<script[^>]*>.*<\/script>/

var old_html = ""
while (html != old_html) {
  old_html = html
  html.replace(script_tag_regex, with: "")
}

...

```
The above sanitizer does not filter out all `<script>` tags. Browsers will not only accept `</script>` as script end tags, but also tags such as `</script foo="bar">` even though it is a parser error. This means that an attack string such as `<script>alert(1)</script foo="bar">` will not be filtered by the function, and `alert(1)` will be executed by a browser if the string is rendered as HTML.

Other corner cases include HTML comments ending with `--!>`, and HTML tag names containing uppercase characters.


## References
* Securitum: [The Curious Case of Copy &amp; Paste](https://research.securitum.com/the-curious-case-of-copy-paste/).
* stackoverflow.com: [You can't parse \[X\]HTML with regex](https://stackoverflow.com/questions/1732348/regex-match-open-tags-except-xhtml-self-contained-tags#answer-1732454).
* HTML Standard: [Comment end bang state](https://html.spec.whatwg.org/multipage/parsing.html#comment-end-bang-state).
* stackoverflow.com: [Why aren't browsers strict about HTML?](https://stackoverflow.com/questions/25559999/why-arent-browsers-strict-about-html)
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
* Common Weakness Enumeration: [CWE-185](https://cwe.mitre.org/data/definitions/185.html).
* Common Weakness Enumeration: [CWE-186](https://cwe.mitre.org/data/definitions/186.html).

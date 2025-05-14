# Suspicious characters in a regular expression
When a character in a string literal or regular expression literal is preceded by a backslash, it is interpreted as part of an escape sequence. For example, the escape sequence `\n` in a string literal corresponds to a single `newline` character, and not the `\` and `n` characters. There are two Go escape sequences that could produce surprising results. First, `regexp.Compile("\a")` matches the bell character, whereas `regexp.Compile("\\A")` matches the start of text and `regexp.Compile("\\a")` is a Vim (but not Go) regular expression matching any alphabetic character. Second, `regexp.Compile("\b")` matches a backspace, whereas `regexp.Compile("\\b")` matches the start of a word. Confusing one for the other could lead to a regular expression passing or failing much more often than expected, with potential security consequences. Note this is less of a problem than in some other languages because in Go, only valid escape sequences are accepted, both in an ordinary string (for example, `s := "\k"` will not compile as there is no such escape sequence) and in regular expressions (for example, `regexp.MustCompile("\\k")` will panic as `\k` does not refer to a character class or other special token according to Go's regular expression grammar).


## Recommendation
Ensure that the right number of backslashes is used when escaping characters in strings and regular expressions.


## Example
The following example code fails to check for a forbidden word in an input string:


```go
package main

import "regexp"

func broken(hostNames []byte) string {
	var hostRe = regexp.MustCompile("\bforbidden.host.org")
	if hostRe.Match(hostNames) {
		return "Must not target forbidden.host.org"
	} else {
		// This will be reached even if hostNames is exactly "forbidden.host.org",
		// because the literal backspace is not matched
		return ""
	}
}

```
The check does not work, but can be fixed by escaping the backslash:


```go
package main

import "regexp"

func fixed(hostNames []byte) string {
	var hostRe = regexp.MustCompile(`\bforbidden.host.org`)
	if hostRe.Match(hostNames) {
		return "Must not target forbidden.host.org"
	} else {
		// hostNames definitely doesn't contain a word "forbidden.host.org", as "\\b"
		// is the start-of-word anchor, not a literal backspace.
		return ""
	}
}

```
Alternatively, you can use backtick-delimited raw string literals. For example, the `\b` in ``` regexp.Compile(`hello\bworld`) ``` matches a word boundary, not a backspace character, as within backticks `\b` is not an escape sequence.


## References
* golang.org: [Overview of the Regexp package](https://golang.org/pkg/regexp/).
* Google: [Syntax of regular expressions accepted by RE2](https://github.com/google/re2/wiki/Syntax).
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).

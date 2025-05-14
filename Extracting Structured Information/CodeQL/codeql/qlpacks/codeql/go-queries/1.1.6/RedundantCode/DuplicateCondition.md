# Duplicate 'if' condition
If two conditions in an 'if'-'else if' chain are identical, the second condition will never hold. This most likely indicates a copy-paste error where the first condition was copied and then not properly adjusted.


## Recommendation
Examine the two conditions to find out what they were meant to check. If both the conditions and the branches that depend on them are identical, then the second branch is duplicate code that can be deleted. Otherwise, the second condition needs to be adjusted.


## Example
In the example below, the function `controller` checks its parameter `msg` to determine what operation it is meant to perform. However, the comparison in the 'else if' is identical to the comparison in the 'if', so this branch will never be taken.


```go
package main

func controller(msg string) {
	if msg == "start" {
		start()
	} else if msg == "start" {
		stop()
	} else {
		panic("Message not understood.")
	}
}

```
Most likely, the 'else if' branch should compare `msg` to `"stop"`:


```go
package main

func controllerGood(msg string) {
	if msg == "start" {
		start()
	} else if msg == "stop" {
		stop()
	} else {
		panic("Message not understood.")
	}
}

```

## References
* The Go Programming Language Specification: [If statements](https://golang.org/ref/spec#If_statements).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).

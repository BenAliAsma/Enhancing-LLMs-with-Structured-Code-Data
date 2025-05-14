# Duplicate switch case
If two cases in a 'switch' statement are identical, the second case will never be executed. This most likely indicates a copy-paste error where the first case was copied and then not properly adjusted.


## Recommendation
Examine the two cases to find out what they were meant to check. If both the conditions and the bodies are identical, then the second case is duplicate code that can be deleted. Otherwise, the second case needs to be adjusted.


## Example
In the example below, the function `controller` checks its parameter `msg` to determine what operation it is meant to perform. However, the condition of the second case is identical to that of the first, so this case will never be executed.


```go
package main

func controller(msg string) {
	switch {
	case msg == "start":
		start()
	case msg == "start":
		stop()
	default:
		panic("Message not understood.")
	}
}

```
Most likely, the second case should compare `msg` to `"stop"`:


```go
package main

func controllerGood(msg string) {
	switch {
	case msg == "start":
		start()
	case msg == "stop":
		stop()
	default:
		panic("Message not understood.")
	}
}

```

## References
* The Go Programming Language Specification: [Switch statements](https://golang.org/ref/spec#Switch_statements).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).

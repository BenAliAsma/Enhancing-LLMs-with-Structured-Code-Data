# Defer in loop
A deferred statement in a loop will not execute until the end of the function. This can lead to unintentionally holding resources open, like file handles or database transactions.


## Recommendation
Either run the deferred function manually, or create a subroutine that contains the defer.


## Example
In the example below, the files opened in the loop are not closed until the end of the function:


```go
package main

import "os"

func openFiles(filenames []string) {
	for _, filename := range filenames {
		file, err := os.Open(filename)
		defer file.Close()
		if err != nil {
			// handle error
		}
		// work on file
	}
}

```
The corrected version puts the loop body into a function.


```go
package main

import "os"

func openFile(filename string) {
	file, err := os.Open(filename)
	defer file.Close()
	if err != nil {
		// handle error
	}
	// work on file
}

func openFilesGood(filenames []string) {
	for _, filename := range filenames {
		openFile(filename)
	}
}

```

## References
* [Defer statements](https://golang.org/ref/spec#Defer_statements).

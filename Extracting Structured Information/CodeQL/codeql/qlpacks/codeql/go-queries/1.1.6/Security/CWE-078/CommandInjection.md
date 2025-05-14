# Command built from user-controlled sources
If a system command invocation is built from user-provided data without sufficient sanitization, a malicious user may be able to run commands to exfiltrate data or compromise the system.


## Recommendation
Whenever possible, use hard-coded string literals for commands and avoid shell string interpreters like `sh -c`.

If given arguments as a single string, avoid simply splitting the string on whitespace. Arguments may contain quoted whitespace, causing them to split into multiple arguments.

If this is not possible, sanitize user input to avoid characters like spaces and various kinds of quotes that can alter the meaning of the command.


## Example
In the following example, assume the function `handler` is an HTTP request handler in a web application, whose parameter `req` contains the request object:


```go
package main

import (
	"net/http"
	"os/exec"
)

func handler(req *http.Request) {
	imageName := req.URL.Query()["imageName"][0]
	outputPath := "/tmp/output.svg"
	cmd := exec.Command("sh", "-c", fmt.Sprintf("imagetool %s > %s", imageName, outputPath))
	cmd.Run()
	// ...
}

```
The handler extracts the image file name from the request and uses the image name to construct a shell command that is executed using ``` `sh -c` ```, which can lead to command injection.

It's better to avoid shell commands by using the `exec.Command` function directly, as shown in the following example:


```go
package main

import (
	"log"
	"net/http"
	"os"
	"os/exec"
)

func handler(req *http.Request) {
	imageName := req.URL.Query()["imageName"][0]
	outputPath := "/tmp/output.svg"

	// Create the output file
	outfile, err := os.Create(outputPath)
	if err != nil {
		log.Fatal(err)
	}
	defer outfile.Close()

	// Prepare the command
	cmd := exec.Command("imagetool", imageName)

	// Set the output to our file
	cmd.Stdout = outfile

	cmd.Run()
}

```
Alternatively, a regular expression can be used to ensure that the image name is safe to use in a shell command:


```go
package main

import (
	"log"
	"net/http"
	"os/exec"
	"regexp"
)

func handler(req *http.Request) {
	imageName := req.URL.Query()["imageName"][0]
	outputPath := "/tmp/output.svg"

	// Validate the imageName with a regular expression
	validImageName := regexp.MustCompile(`^[a-zA-Z0-9_\-\.]+$`)
	if !validImageName.MatchString(imageName) {
		log.Fatal("Invalid image name")
		return
	}

	cmd := exec.Command("sh", "-c", fmt.Sprintf("imagetool %s > %s", imageName, outputPath))
	cmd.Run()
}

```
Some commands, like `git`, can indirectly execute commands if an attacker specifies the flags given to the command.

To mitigate this risk, either add a `--` argument to ensure subsequent arguments are not interpreted as flags, or verify that the argument does not start with `"--"`.


```go
package main

import (
	"log"
	"net/http"
	"os/exec"
	"strings"
)

func handler(req *http.Request) {
	repoURL := req.URL.Query()["repoURL"][0]
	outputPath := "/tmp/repo"

	// Sanitize the repoURL to ensure it does not start with "--"
	if strings.HasPrefix(repoURL, "--") {
		log.Fatal("Invalid repository URL")
	} else {
		cmd := exec.Command("git", "clone", repoURL, outputPath)
		err := cmd.Run()
		if err != nil {
			log.Fatal(err)
		}
	}

	// Or: add "--" to ensure that the repoURL is not interpreted as a flag
	cmd := exec.Command("git", "clone", "--", repoURL, outputPath)
	err := cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
}

```

## References
* OWASP: [Command Injection](https://www.owasp.org/index.php/Command_Injection).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).

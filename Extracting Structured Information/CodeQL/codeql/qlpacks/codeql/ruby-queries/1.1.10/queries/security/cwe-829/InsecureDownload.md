# Download of sensitive file through insecure connection
Downloading executables or other sensitive files over an unencrypted connection can leave a server open to man-in-the-middle attacks (MITM). Such an attack can allow an attacker to insert arbitrary content into the downloaded file, and in the worst case, allow the attacker to execute arbitrary code on the vulnerable system.


## Recommendation
Use a secure transfer protocol when downloading executables or other sensitive files.


## Example
In this example, a server downloads a shell script from a remote URL and then executes the script.


```ruby
require "net/http"

script = Net::HTTP.new("http://mydownload.example.org").get("/myscript.sh").body
system(script)
```
The HTTP protocol is vulnerable to MITM, and thus an attacker could potentially replace the downloaded shell script with arbitrary code, which gives the attacker complete control over the system.

The issue has been fixed in the example below by replacing the HTTP protocol with the HTTPS protocol.


```ruby
require "net/http"

script = Net::HTTP.new("https://mydownload.example.org").get("/myscript.sh").body
system(script)
```

## References
* Wikipedia: [Man-in-the-middle attack](https://en.wikipedia.org/wiki/Man-in-the-middle_attack)
* Common Weakness Enumeration: [CWE-829](https://cwe.mitre.org/data/definitions/829.html).

# Log injection
If unsanitized user input is written to a log entry, a malicious user may able to forge new log entries.

Forgery can occur if a user provides some input with characters that are interpreted when the log output is displayed. If the log is displayed as a plain text file, then new line characters can be used by a malicious user. If the log is displayed as HTML, then arbitrary HTML may be included to spoof log entries.


## Recommendation
User input should be suitably sanitized before it is logged. Suitable means of sanitization depend on how the log entries will be displayed or consumed.

If the log entries are in plain text then line breaks should be removed from user input, using `String#gsub` or similar. Care should also be taken that user input is clearly marked in log entries.

For log entries that will be displayed in HTML, user input should be HTML-encoded before being logged, to prevent forgery and other forms of HTML injection.


## Example
In the example, a username, provided by the user, is logged using \`Logger\#info\`.

In the first case, it is logged without any sanitization. If a malicious user provides \`username=Guest%0a\[INFO\]+User:+Admin%0a\` as a username parameter, the log entry will be split in two different lines, where the second line will be \`\[INFO\]+User:+Admin\`.


```ruby
require 'logger'

class UsersController < ApplicationController
  def login
    logger = Logger.new STDOUT
    username = params[:username]

    # BAD: log message constructed with unsanitized user input
    logger.info "attempting to login user: " + username

    # ... login logic ...
  end
end

```
In the second example, `String#gsub` is used to ensure no line endings are present in the user input.


```ruby
require 'logger'

class UsersController < ApplicationController
  def login
    logger = Logger.new STDOUT
    username = params[:username]

    # GOOD: log message constructed with sanitized user input
    logger.info "attempting to login user: " + sanitized_username.gsub("\n", "")

    # ... login logic ...
  end
end

```

## References
* OWASP: [Log Injection](https://www.owasp.org/index.php/Log_Injection).
* Common Weakness Enumeration: [CWE-117](https://cwe.mitre.org/data/definitions/117.html).

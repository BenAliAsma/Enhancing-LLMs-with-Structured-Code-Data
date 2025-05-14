# Server crash
Servers handle requests from clients until terminated deliberately by a server administrator. A client request that results in an uncaught server-side exception causes the current server response generation to fail, and should not have an effect on subsequent client requests.

Under some circumstances, uncaught exceptions can however cause the entire server to terminate abruptly. Such a behavior is highly undesirable, especially if it gives malicious users the ability to turn off the server at will, which is an efficient denial-of-service attack.


## Recommendation
Ensure that the processing of client requests can not cause uncaught exceptions to terminate the entire server abruptly.


## Example
The following server code checks if a client-provided file path is valid before saving data to that path. It would be reasonable to expect that the server responds with an error in case the request contains an invalid file path. However, the server instead throws an exception, which is uncaught in the context of the asynchronous callback invocation (`fs.access(...)`). This causes the entire server to terminate abruptly.


```javascript
const express = require("express"),
  fs = require("fs");

function save(rootDir, path, content) {
  if (!isValidPath(rootDir, req.query.filePath)) {
    throw new Error(`Invalid filePath: ${req.query.filePath}`); // BAD crashes the server
  }
  // write content to disk
}

express().post("/save", (req, res) => {
  fs.access(rootDir, (err) => {
    if (err) {
      console.error(
        `Server setup is corrupted, ${rootDir} cannot be accessed!`
      );
      res.status(500);
      res.end();
      return;
    }
    save(rootDir, req.query.path, req.body);
    res.status(200);
    res.end();
  });
});

```
To remedy this, the server can catch the exception explicitly with a `try/catch` block, and generate an appropriate error response instead:


```javascript
// ...
express().post("/save", (req, res) => {
  fs.access(rootDir, (err) => {
    // ...
    try {
      save(rootDir, req.query.path, req.body); // GOOD exception is caught below
      res.status(200);
      res.end();
    } catch (e) {
      res.status(500);
      res.end();
    }
  });
});

```
To simplify exception handling, it may be advisable to switch to async/await syntax instead of using callbacks, which allows wrapping the entire request handler in a `try/catch` block:


```javascript
// ...
express().post("/save", async (req, res) => {
  try {
    await fs.promises.access(rootDir);
    save(rootDir, req.query.path, req.body); // GOOD exception is caught below
    res.status(200);
    res.end();
  } catch (e) {
    res.status(500);
    res.end();
  }
});

```

## References
* Common Weakness Enumeration: [CWE-248](https://cwe.mitre.org/data/definitions/248.html).
* Common Weakness Enumeration: [CWE-730](https://cwe.mitre.org/data/definitions/730.html).

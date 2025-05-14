# Resource exhaustion with additional heuristic sources
Applications are constrained by how many resources they can make use of. Failing to respect these constraints may cause the application to be unresponsive or crash. It is therefore problematic if attackers can control the sizes or lifetimes of allocated objects.


## Recommendation
Ensure that attackers can not control object sizes and their lifetimes. If object sizes and lifetimes must be controlled by external parties, ensure you restrict the object sizes and lifetimes so that they are within acceptable ranges.


## Example
The following example allocates a buffer with a user-controlled size.


```javascript
var http = require("http"),
    url = require("url");

var server = http.createServer(function(req, res) {
	var size = parseInt(url.parse(req.url, true).query.size);

	let buffer = Buffer.alloc(size); // BAD

	// ... use the buffer
});
```
This is problematic since an attacker can choose a size that makes the application run out of memory. Even worse, in older versions of Node.js, this could leak confidential memory. To prevent such attacks, limit the buffer size:


```javascript
var http = require("http"),
    url = require("url");

var server = http.createServer(function(req, res) {
	var size = parseInt(url.parse(req.url, true).query.size);

	if (size > 1024) {
		res.statusCode = 400;
		res.end("Bad request.");
		return;
	}

	let buffer = Buffer.alloc(size); // GOOD

	// ... use the buffer
});
```

## Example
As another example, consider an application that allocates an array with a user-controlled size, and then fills it with values:


```javascript
var http = require("http"),
    url = require("url");

var server = http.createServer(function(req, res) {
	var size = parseInt(url.parse(req.url, true).query.size);

	let dogs = new Array(size).fill("dog"); // BAD

	// ... use the dog
});
```
The allocation of the array itself is not problematic since arrays are allocated sparsely, but the subsequent filling of the array will take a long time, causing the application to be unresponsive, or even run out of memory. Again, a limit on the size will prevent the attack:


```javascript
var http = require("http"),
    url = require("url");

var server = http.createServer(function(req, res) {
	var size = parseInt(url.parse(req.url, true).query.size);

	if (size > 1024) {
		res.statusCode = 400;
		res.end("Bad request.");
		return;
	}

	let dogs = new Array(size).fill("dog"); // GOOD

	// ... use the dogs
});
```

## Example
Finally, the following example lets a user choose a delay after which a function is executed:


```javascript
var http = require("http"),
    url = require("url");

var server = http.createServer(function(req, res) {
	var delay = parseInt(url.parse(req.url, true).query.delay);

	setTimeout(f, delay); // BAD

});

```
This is problematic because a large delay essentially makes the application wait indefinitely before executing the function. Repeated registrations of such delays will therefore use up all of the memory in the application. A limit on the delay will prevent the attack:


```javascript
var http = require("http"),
    url = require("url");

var server = http.createServer(function(req, res) {
	var delay = parseInt(url.parse(req.url, true).query.delay);

	if (delay > 1000) {
		res.statusCode = 400;
		res.end("Bad request.");
		return;
	}

	setTimeout(f, delay); // GOOD

});

```

## References
* Wikipedia: [Denial-of-service attack](https://en.wikipedia.org/wiki/Denial-of-service_attack).
* Common Weakness Enumeration: [CWE-400](https://cwe.mitre.org/data/definitions/400.html).
* Common Weakness Enumeration: [CWE-770](https://cwe.mitre.org/data/definitions/770.html).

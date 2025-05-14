# TODO/FIXME comments
A comment that includes the word `TODO` or `FIXME` often marks a part of the code that is incomplete or broken, or highlights ambiguities in the software's specification.

For example, this list of comments is typical of those found in real programs:

* `TODO: move this code somewhere else`
* `FIXME: handle this case`
* `FIXME: find a better solution to this workaround`
* `TODO: test this`

## Recommendation
It is very important that `TODO` or `FIXME` comments are not just removed from the code. Each of them must be addressed in some way.

Simpler comments can usually be immediately addressed by fixing the code, adding a test, doing some refactoring, or clarifying the intended behavior of a feature.

In contrast, larger issues may require discussion, and a significant amount of work to address. In these cases it is a good idea to move the comment to an issue-tracking system, so that the issue can be tracked and prioritized relative to other defects and feature requests.


## References
* Approxion: [TODO or not TODO](http://www.approxion.com/?p=39).
* Wikipedia: [Comment tags](http://en.wikipedia.org/wiki/Comment_%28computer_programming%29#Tags), [Issue tracking system](http://en.wikipedia.org/wiki/Issue_tracking_system).
* Common Weakness Enumeration: [CWE-546](https://cwe.mitre.org/data/definitions/546.html).

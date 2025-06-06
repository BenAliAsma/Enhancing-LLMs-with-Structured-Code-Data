/**
 * @name Untrusted data passed to external API
 * @description Data provided remotely is used in this external API without sanitization, which could be a security risk.
 * @id js/untrusted-data-to-external-api
 * @kind path-problem
 * @precision low
 * @problem.severity error
 * @security-severity 7.8
 * @tags security external/cwe/cwe-20
 */

import javascript
import semmle.javascript.security.dataflow.ExternalAPIUsedWithUntrustedDataQuery
import ExternalAPIUsedWithUntrustedDataFlow::PathGraph

from
  ExternalAPIUsedWithUntrustedDataFlow::PathNode source,
  ExternalAPIUsedWithUntrustedDataFlow::PathNode sink
where ExternalAPIUsedWithUntrustedDataFlow::flowPath(source, sink)
select sink, source, sink,
  "Call to " + sink.getNode().(Sink).getApiName() + " with untrusted data from $@.", source,
  source.toString()

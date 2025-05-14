# Complex condition
This rule finds boolean expressions that have more than 5 consecutive operators that are not of the same type (e.g. alternating `&&` and `||` operators). Long chains of operators of the same type are not flagged as violations of this rule.

Complex boolean expressions are hard to read. Consequently, when modifying such expressions there is an increased risk of introducing defects. Naming intermediate results as local variables will make the logic easier to read and understand.


## Recommendation
Use local variables or macros to represent intermediate values to make the condition easier to understand.


## Example

```cpp
//This condition is too complex and can be improved by using local variables
bool accept_message =
	(message_type == CONNECT && _state != CONNECTED) ||
	(message_type == DISCONNECT && _state == CONNECTED) ||
	(message_type == DATA && _state == CONNECTED);

//This condition is acceptable, as all the logical operators are of the same type (&&)
bool valid_connect =
	message_type == CONNECT && 
	_state != CONNECTED &&
	time_since_prev_connect > MAX_CONNECT_INTERVAL &&
	message_length <= MAX_PACKET_SIZE &&
	checksum(message) == get_checksum_field(message);
```

## References
* [Operators](http://www.cplusplus.com/doc/tutorial/operators/)
* [Conditionals](http://geosoft.no/development/cppstyle.html#Conditionals)

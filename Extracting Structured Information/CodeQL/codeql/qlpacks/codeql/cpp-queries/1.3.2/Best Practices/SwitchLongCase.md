# Long switch case
This rule finds switch statements that have too much code in their cases. Long case statements often lead to large amounts of nesting, adding to the difficulty of understanding what the code actually does. Consider wrapping the code for each case in a function and just using the switch statement to invoke the appropriate function in each case.

The indicated switch statement has a case that is more than 30 lines long.


## Recommendation
Consider creating a separate function for the code in the long case statement.


## Example

```cpp
//This switch statement has long case statements, and can become difficult to
//read as the processing for each message type becomes more complex
switch (message_type) {
	case CONNECT:
		_state = CONNECTING;
		int message_id = message_get_id(message);
		int source = connect_get_source(message);
		//More code here...
		send(connect_response);
		break;
	case DISCONNECT:
		_state = DISCONNECTING;
		int message_id = message_get_id(message);
		int source = disconnect_get_source(message);
		//More code here...
		send(disconnect_response);
		break;
	default:
		log("Invalid message, id : %d", message_get_id(message));
}

//This is better, as each case is split out to a separate function
switch (packet_type) {
	case STREAM:
		process_stream_packet(packet);
		break;
	case DATAGRAM:
		process_datagram_packet(packet);
		break;
	default:
		log("Invalid packet type: %d", packet_type);
}
```

## References
* [Switch statements](http://www.learncpp.com/cpp-tutorial/53-switch-statements/)

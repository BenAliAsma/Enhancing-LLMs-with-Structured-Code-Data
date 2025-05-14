# Number of events
This metric counts the number of events each reference type has. If a reference type has a large number of events then it is possible the events are too detailed or the class has too many responsibilities.


## Recommendation
Consider whether the class is responsible for more than one thing. If it is then it may be best to split the class in two. If the class only has one responsibility and it still has too many events then you should also consider whether the detail provided by the different events could instead be provided by the `EventArgs`.


## Example
In this example the class has a lot of events.


```csharp
class HTTPRequest
{
    public class HTTPRequestSentArgs : EventArgs { }
    public class HTTPRequestResponseStartArgs : EventArgs { }
    public class HTTPRequestResponseEndArgs : EventArgs { }
    public class HTTPRequestResponseProcessedArgs : EventArgs { }

    public delegate void HTTPRequestSent(object sender, HTTPRequestSentArgs e);
    public delegate void HTTPRequestResponseStart(object sender, HTTPRequestResponseStartArgs e);
    public delegate void HTTPRequestResponseEnd(object sender, HTTPRequestResponseEndArgs e);
    public delegate void HTTPRequestResponseProcessed(object sender, HTTPRequestResponseProcessedArgs e);

    public event HTTPRequestSent HTTPRequestSentEvent;
    public event HTTPRequestResponseStart HTTPRequestResponseStartEvent;
    public event HTTPRequestResponseEnd HTTPRequestResponseEndEvent;
    public event HTTPRequestResponseProcessed HTTPRequestResponseProcessedEvent;

    public void send()
    {
        HTTPRequestSentEvent(this, new HTTPRequestSentArgs());
        HTTPRequestResponseStartEvent(this, new HTTPRequestResponseStartArgs());
        HTTPRequestResponseEndEvent(this, new HTTPRequestResponseEndArgs());
        HTTPRequestResponseProcessedEvent(this, new HTTPRequestResponseProcessedArgs());
    }
}

```

## Fix
One possible improvement is to provide the detail of the event in a property of the event arguments. If the event arguments are very similar for two different events then this is a key sign that using a single event might be preferable.


```csharp
class HTTPRequest
{
    public class HTTPRequestStateChangeArgs : EventArgs
    {
        private String state;
        public HTTPRequestStateChangeArgs(String state)
        {
            this.state = state;
        }
    }

    public delegate void HTTPRequestStateChange(object sender, HTTPRequestStateChangeArgs e);

    public event HTTPRequestStateChange HTTPRequestStateChangeEvent;

    public void send()
    {
        HTTPRequestStateChangeEvent(this, new HTTPRequestStateChangeArgs("sent"));
        HTTPRequestStateChangeEvent(this, new HTTPRequestStateChangeArgs("responsestart"));
        HTTPRequestStateChangeEvent(this, new HTTPRequestStateChangeArgs("responseend"));
        HTTPRequestStateChangeEvent(this, new HTTPRequestStateChangeArgs("responseprocessed"));
    }
}

```

## References
* MSDN. C\# Programmer's Reference. [Events Tutorial](http://msdn.microsoft.com/en-us/library/aa645739(v=vs.71).aspx).
